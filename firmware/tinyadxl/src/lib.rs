//! Minimal ADXL345 driver over 4-wire SPI, `no_std`, for `embedded-hal` 1.0 `SpiDevice`.
//!
//! Configures the chip for vibration sensing (full resolution, +-16 g, 3.9 mg per count, stream-mode
//! FIFO with a watermark interrupt) and reads samples out of the 32-entry FIFO.
//!
//! **Status: tested only against a simulated chip (`tests/sim.rs`); it has not touched a real ADXL345.**
//! The register map, the bit meanings, the command byte (`R/W` bit 7, `MB` bit 6), the 5 MHz limit, the
//! output-rate codes and the FIFO timing rule below were cross-checked against two independent code bases
//! (Linux's `adxl34x` driver and Adafruit's `Adafruit_ADXL345`), not against the datasheet PDF, which could
//! not be fetched. Still unverified: the required SPI mode (the datasheet is believed to ask for mode 3, idle
//! clock high, sample on the second edge), the 3.9 mg per count scale, and everything about real timing.
//! The bench items are in `docs/bringup-checklist.md`.
//!
//! **FIFO read timing.** After a FIFO data read there must be at least 5 microseconds, with chip select
//! released, before the next FIFO read or a `FIFO_STATUS` read (Linux's driver quotes this from the datasheet;
//! at SPI clocks of 1.5 MHz or less the addressing bytes already provide it, above that chip select must be
//! deasserted). [`Adxl345::read_fifo`] takes a delay provider and leaves that gap after every sample;
//! [`Adxl345::read_sample`] does not, so the caller must wait before touching the FIFO again.

#![no_std]

use embedded_hal::delay::DelayNs;
use embedded_hal::spi::Operation;
use embedded_hal::spi::SpiDevice;

pub const DEVICE_ID: u8 = 0xE5;
/// Full-resolution scale: 3.9 mg per count at every range.
pub const MG_PER_COUNT_X10: i32 = 39;
/// The FIFO holds 32 samples.
pub const FIFO_DEPTH: usize = 32;

mod reg {
    pub const DEVID: u8 = 0x00;
    pub const BW_RATE: u8 = 0x2C;
    pub const POWER_CTL: u8 = 0x2D;
    pub const INT_ENABLE: u8 = 0x2E;
    pub const INT_SOURCE: u8 = 0x30;
    pub const DATA_FORMAT: u8 = 0x31;
    pub const DATAX0: u8 = 0x32;
    pub const FIFO_CTL: u8 = 0x38;
    pub const FIFO_STATUS: u8 = 0x39;
}

const READ: u8 = 0x80;
const MULTI: u8 = 0x40;
const POWER_MEASURE: u8 = 0x08;
const DATA_FORMAT_FULL_RES: u8 = 0x08;
const FIFO_MODE_STREAM: u8 = 0x80;
const INT_WATERMARK: u8 = 0x02;
const INT_OVERRUN: u8 = 0x01;
/// At least 5 microseconds, chip select released, after a FIFO read.
const FIFO_READ_GAP_US: u32 = 5;

/// Output data rate. The code is the low nibble of `BW_RATE`; bandwidth is half the rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Odr {
    Hz100 = 0x0A,
    Hz200 = 0x0B,
    Hz400 = 0x0C,
    Hz800 = 0x0D,
    Hz1600 = 0x0E,
    Hz3200 = 0x0F,
}

impl Odr {
    pub const fn hz(self) -> u32 {
        match self {
            Odr::Hz100 => 100,
            Odr::Hz200 => 200,
            Odr::Hz400 => 400,
            Odr::Hz800 => 800,
            Odr::Hz1600 => 1600,
            Odr::Hz3200 => 3200,
        }
    }
}

/// Measurement range (`DATA_FORMAT[1:0]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    G2 = 0,
    G4 = 1,
    G8 = 2,
    G16 = 3,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error<E> {
    Spi(E),
    /// `DEVID` was not 0xE5 (wiring, chip select or a different part).
    WrongDevice(u8),
}

pub struct Adxl345<SPI> {
    spi: SPI,
}

impl<SPI: SpiDevice> Adxl345<SPI> {
    pub fn new(spi: SPI) -> Self {
        Self { spi }
    }

    pub fn release(self) -> SPI {
        self.spi
    }

    /// The underlying bus, for tests and for sharing it with other devices.
    pub fn bus_mut(&mut self) -> &mut SPI {
        &mut self.spi
    }

    fn read_regs(&mut self, addr: u8, buf: &mut [u8]) -> Result<(), Error<SPI::Error>> {
        let cmd = [READ | MULTI | addr];
        self.spi
            .transaction(&mut [Operation::Write(&cmd), Operation::Read(buf)])
            .map_err(Error::Spi)
    }

    fn write_reg(&mut self, addr: u8, value: u8) -> Result<(), Error<SPI::Error>> {
        self.spi.write(&[addr & 0x3F, value]).map_err(Error::Spi)
    }

    fn read_reg(&mut self, addr: u8) -> Result<u8, Error<SPI::Error>> {
        let mut b = [0u8];
        self.read_regs(addr, &mut b)?;
        Ok(b[0])
    }

    /// Check the device ID, then configure: output rate, full-resolution `range`, stream FIFO with the
    /// given watermark (1..=31 samples), watermark and overrun interrupts, and start measuring.
    pub fn init(&mut self, odr: Odr, range: Range, watermark: u8) -> Result<(), Error<SPI::Error>> {
        let id = self.read_reg(reg::DEVID)?;
        if id != DEVICE_ID {
            return Err(Error::WrongDevice(id));
        }
        self.write_reg(reg::POWER_CTL, 0x00)?; // standby while configuring
        self.write_reg(reg::BW_RATE, odr as u8)?;
        self.write_reg(reg::DATA_FORMAT, DATA_FORMAT_FULL_RES | range as u8)?;
        self.write_reg(
            reg::FIFO_CTL,
            FIFO_MODE_STREAM | (watermark.clamp(1, 31) & 0x1F),
        )?;
        self.write_reg(reg::INT_ENABLE, INT_WATERMARK | INT_OVERRUN)?;
        self.write_reg(reg::POWER_CTL, POWER_MEASURE)
    }

    /// Samples waiting in the FIFO (0..=32).
    pub fn fifo_level(&mut self) -> Result<usize, Error<SPI::Error>> {
        Ok((self.read_reg(reg::FIFO_STATUS)? & 0x3F) as usize)
    }

    /// Reading `INT_SOURCE` clears it. Returns (watermark reached, overrun happened).
    pub fn interrupt_source(&mut self) -> Result<(bool, bool), Error<SPI::Error>> {
        let s = self.read_reg(reg::INT_SOURCE)?;
        Ok((s & INT_WATERMARK != 0, s & INT_OVERRUN != 0))
    }

    /// One sample (x, y, z) in raw counts, popped from the FIFO. 3.9 mg per count. Leaves no gap afterwards:
    /// wait 5 microseconds before the next FIFO access (see the module notes), or use [`Adxl345::read_fifo`].
    pub fn read_sample(&mut self) -> Result<[i16; 3], Error<SPI::Error>> {
        let mut b = [0u8; 6];
        self.read_regs(reg::DATAX0, &mut b)?;
        Ok([
            i16::from_le_bytes([b[0], b[1]]),
            i16::from_le_bytes([b[2], b[3]]),
            i16::from_le_bytes([b[4], b[5]]),
        ])
    }

    /// Pop up to `out.len()` samples from the FIFO, oldest first; returns how many were read (at most the
    /// FIFO level when called). After each read, chip select is released and `delay` waits 5 microseconds,
    /// so the next read, and a following `fifo_level`, respect the datasheet's FIFO timing rule.
    pub fn read_fifo<D: DelayNs>(
        &mut self,
        delay: &mut D,
        out: &mut [[i16; 3]],
    ) -> Result<usize, Error<SPI::Error>> {
        let n = self.fifo_level()?.min(out.len());
        for slot in out.iter_mut().take(n) {
            *slot = self.read_sample()?;
            delay.delay_us(FIFO_READ_GAP_US);
        }
        Ok(n)
    }
}

/// Raw full-resolution counts to milli-g (3.9 mg per count), rounded toward zero.
pub const fn to_milli_g(raw: i16) -> i32 {
    raw as i32 * MG_PER_COUNT_X10 / 10
}
