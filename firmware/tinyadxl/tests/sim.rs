//! The driver against a simulated ADXL345.
//!
//! The simulator models the parts of the chip the driver touches: the register file, the 32-entry stream
//! FIFO (reading `DATAX0` pops one sample), `FIFO_STATUS`, and `INT_SOURCE` (clear on read). It rejects
//! transfers the real chip's framing would not allow, and enforces the FIFO timing rule: at least 5
//! microseconds (on a simulated clock that only `SimDelay` advances) between a FIFO pop and the next FIFO
//! or `FIFO_STATUS` access. It was written from the same reading of the chip as the driver, so passing
//! shows the driver is consistent with that reading, not that the reading is right.

use std::cell::Cell;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::rc::Rc;

use embedded_hal::delay::DelayNs;
use embedded_hal::spi::{ErrorType, Operation, SpiDevice};
use tinyadxl::{to_milli_g, Adxl345, Error, Odr, Range, DEVICE_ID, FIFO_DEPTH};

const DEVID: usize = 0x00;
const BW_RATE: usize = 0x2C;
const POWER_CTL: usize = 0x2D;
const INT_ENABLE: usize = 0x2E;
const INT_SOURCE: usize = 0x30;
const DATA_FORMAT: usize = 0x31;
const DATAX0: usize = 0x32;
const FIFO_CTL: usize = 0x38;
const FIFO_STATUS: usize = 0x39;
const FIFO_GAP_NS: u64 = 5_000;

struct Sim {
    regs: [u8; 64],
    fifo: VecDeque<[i16; 3]>,
    overrun: bool,
    headers: Vec<u8>,
    measuring: bool,
    clock_ns: Rc<Cell<u64>>,
    last_pop_ns: Option<u64>,
}

struct SimDelay {
    clock_ns: Rc<Cell<u64>>,
}

impl DelayNs for SimDelay {
    fn delay_ns(&mut self, ns: u32) {
        self.clock_ns.set(self.clock_ns.get() + ns as u64);
    }
}

impl Sim {
    fn new() -> (Self, SimDelay) {
        let clock = Rc::new(Cell::new(0));
        let mut regs = [0u8; 64];
        regs[DEVID] = DEVICE_ID;
        (
            Self {
                regs,
                fifo: VecDeque::new(),
                overrun: false,
                headers: vec![],
                measuring: false,
                clock_ns: clock.clone(),
                last_pop_ns: None,
            },
            SimDelay { clock_ns: clock },
        )
    }

    fn push_sample(&mut self, s: [i16; 3]) {
        if !self.measuring {
            return; // a chip in standby does not sample
        }
        if self.fifo.len() == FIFO_DEPTH {
            self.fifo.pop_front();
            self.overrun = true;
        }
        self.fifo.push_back(s);
    }

    fn watermark_reached(&self) -> bool {
        self.fifo.len() >= (self.regs[FIFO_CTL] & 0x1F) as usize
    }

    /// FIFO pops and FIFO_STATUS reads need 5 us since the last pop.
    fn check_fifo_gap(&self, what: &str) {
        if let Some(t) = self.last_pop_ns {
            let waited = self.clock_ns.get() - t;
            assert!(
                waited >= FIFO_GAP_NS,
                "{what} only {waited} ns after a FIFO read; the chip needs {FIFO_GAP_NS} ns"
            );
        }
    }
}

impl ErrorType for Sim {
    type Error = Infallible;
}

impl SpiDevice for Sim {
    fn transaction(&mut self, ops: &mut [Operation<'_, u8>]) -> Result<(), Infallible> {
        let mut written: Vec<u8> = vec![];
        let mut read_addr: Option<usize> = None;
        let mut pending: VecDeque<u8> = VecDeque::new();
        for op in ops.iter_mut() {
            match op {
                Operation::Write(w) => written.extend_from_slice(w),
                Operation::DelayNs(_) => panic!(
                    "a delay inside a transaction keeps chip select low; use the delay provider"
                ),
                Operation::Read(buf) => {
                    let header = written[0];
                    assert!(header & 0x80 != 0, "a read transaction must have R/W = 1");
                    if buf.len() > 1 {
                        assert!(header & 0x40 != 0, "multi-byte read without the MB bit");
                    }
                    let addr = read_addr.get_or_insert((header & 0x3F) as usize);
                    for slot in buf.iter_mut() {
                        if pending.is_empty() {
                            match *addr {
                                DATAX0 => {
                                    self.check_fifo_gap("a FIFO read");
                                    let s = self.fifo.pop_front().unwrap_or([0; 3]);
                                    for v in s {
                                        pending.extend(v.to_le_bytes());
                                    }
                                    self.last_pop_ns = Some(self.clock_ns.get());
                                }
                                FIFO_STATUS => {
                                    self.check_fifo_gap("a FIFO_STATUS read");
                                    pending.push_back(self.fifo.len() as u8)
                                }
                                INT_SOURCE => {
                                    let v =
                                        (self.watermark_reached() as u8) << 1 | self.overrun as u8;
                                    self.overrun = false;
                                    pending.push_back(v);
                                }
                                a => pending.push_back(self.regs[a]),
                            }
                        }
                        *slot = pending.pop_front().unwrap();
                    }
                    self.headers.push(header);
                }
                _ => unreachable!(),
            }
        }
        if read_addr.is_none() {
            // a write: header is the register address (R/W = 0, MB = 0), then one value
            assert_eq!(written.len(), 2, "register writes here are one byte");
            assert!(written[0] & 0xC0 == 0, "write with R/W or MB set");
            let reg = (written[0] & 0x3F) as usize;
            self.regs[reg] = written[1];
            if reg == POWER_CTL {
                self.measuring = written[1] & 0x08 != 0;
            }
            self.headers.push(written[0]);
        }
        Ok(())
    }
}

fn started() -> (Adxl345<Sim>, SimDelay) {
    let (sim, delay) = Sim::new();
    let mut d = Adxl345::new(sim);
    d.init(Odr::Hz3200, Range::G16, 16).unwrap();
    (d, delay)
}

#[test]
fn init_writes_the_expected_configuration() {
    let sim = started().0.release();
    assert_eq!(sim.regs[BW_RATE], 0x0F, "3200 Hz");
    assert_eq!(sim.regs[DATA_FORMAT], 0x0B, "full resolution, +-16 g");
    assert_eq!(sim.regs[FIFO_CTL], 0x90, "stream mode, watermark 16");
    assert_eq!(sim.regs[INT_ENABLE], 0x03, "watermark and overrun");
    assert_eq!(sim.regs[POWER_CTL], 0x08, "measuring");
}

#[test]
fn a_wrong_device_id_is_reported_and_nothing_is_configured() {
    let (mut sim, _) = Sim::new();
    sim.regs[DEVID] = 0x12;
    let mut d = Adxl345::new(sim);
    assert_eq!(
        d.init(Odr::Hz800, Range::G8, 16),
        Err(Error::WrongDevice(0x12))
    );
    assert_eq!(
        d.release().regs[BW_RATE],
        0,
        "must not write registers after a failed ID check"
    );
}

#[test]
fn watermark_is_clamped_to_what_five_bits_can_hold() {
    let (sim, _) = Sim::new();
    let mut d = Adxl345::new(sim);
    d.init(Odr::Hz100, Range::G2, 40).unwrap();
    assert_eq!(d.release().regs[FIFO_CTL] & 0x1F, 31);
}

#[test]
fn samples_are_little_endian_and_sign_extended() {
    let (mut d, _) = started();
    d.bus_mut().push_sample([-4096, 4095, -1]);
    assert_eq!(d.read_sample().unwrap(), [-4096, 4095, -1]);
}

#[test]
fn fifo_level_read_order_and_empty_behaviour() {
    let (mut d, mut delay) = started();
    for i in 0..20 {
        d.bus_mut().push_sample([i, -i, 2 * i]);
    }
    assert_eq!(d.fifo_level().unwrap(), 20);
    let mut buf = [[0i16; 3]; 8];
    assert_eq!(d.read_fifo(&mut delay, &mut buf).unwrap(), 8);
    assert_eq!(buf[0], [0, 0, 0]);
    assert_eq!(buf[7], [7, -7, 14]);
    let mut rest = [[0i16; 3]; 32];
    assert_eq!(
        d.read_fifo(&mut delay, &mut rest).unwrap(),
        12,
        "only what is there"
    );
    assert_eq!(rest[0], [8, -8, 16]);
    assert_eq!(rest[11], [19, -19, 38]);
    assert_eq!(d.read_fifo(&mut delay, &mut rest).unwrap(), 0);
    assert_eq!(d.fifo_level().unwrap(), 0);
}

#[test]
fn read_fifo_leaves_the_five_microsecond_gap_with_chip_select_released() {
    // the simulator panics on any FIFO or FIFO_STATUS access sooner than 5 us after a pop, and on a delay
    // inside a transaction; draining 32 samples and polling the level afterwards must therefore pass
    let (mut d, mut delay) = started();
    for i in 0..32 {
        d.bus_mut().push_sample([i, 0, 0]);
    }
    let mut buf = [[0i16; 3]; 32];
    assert_eq!(d.read_fifo(&mut delay, &mut buf).unwrap(), 32);
    assert_eq!(d.fifo_level().unwrap(), 0);
}

#[test]
#[should_panic(expected = "the chip needs 5000 ns")]
fn the_simulator_catches_back_to_back_fifo_reads() {
    let (mut d, _) = started();
    d.bus_mut().push_sample([1, 2, 3]);
    d.bus_mut().push_sample([4, 5, 6]);
    d.read_sample().unwrap();
    d.read_sample().unwrap(); // no delay in between
}

#[test]
fn overrun_and_watermark_are_reported_and_overrun_clears_on_read() {
    let (mut d, mut delay) = started();
    for i in 0..(FIFO_DEPTH as i16 + 3) {
        d.bus_mut().push_sample([i, 0, 0]);
    }
    assert_eq!(d.fifo_level().unwrap(), FIFO_DEPTH);
    assert_eq!(d.interrupt_source().unwrap(), (true, true));
    assert_eq!(
        d.interrupt_source().unwrap(),
        (true, false),
        "overrun cleared, watermark still met"
    );
    let mut first = [[0i16; 3]; 1];
    d.read_fifo(&mut delay, &mut first).unwrap();
    assert_eq!(first[0][0], 3, "the three oldest samples were overwritten");
}

#[test]
fn framing_bytes_are_right() {
    let (mut d, _) = started();
    d.bus_mut().push_sample([1, 2, 3]);
    d.read_sample().unwrap();
    let headers = d.release().headers;
    assert!(
        headers.contains(&0xF2),
        "multi-byte read of DATAX0 is 0x80 | 0x40 | 0x32"
    );
    assert!(headers.contains(&0x2C), "BW_RATE write has R/W = 0");
}

#[test]
fn counts_convert_to_milli_g_at_3_9_mg_per_count() {
    assert_eq!(to_milli_g(0), 0);
    assert_eq!(to_milli_g(256), 998); // 256 counts is about 1 g
    assert_eq!(to_milli_g(-256), -998);
    assert_eq!(to_milli_g(4095), 15_970);
}

#[test]
fn odr_codes_and_rates() {
    assert_eq!(Odr::Hz3200 as u8, 0x0F);
    assert_eq!(Odr::Hz100 as u8, 0x0A);
    assert_eq!(
        [
            Odr::Hz100,
            Odr::Hz200,
            Odr::Hz400,
            Odr::Hz800,
            Odr::Hz1600,
            Odr::Hz3200
        ]
        .map(Odr::hz),
        [100, 200, 400, 800, 1600, 3200]
    );
}
