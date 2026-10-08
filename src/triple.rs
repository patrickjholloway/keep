//! triple.rs — a lock-free, allocation-free triple buffer (one writer thread, one reader thread).
//!
//! The render thread publishes one surface descriptor per video frame; the audio callback wants
//! the newest one whenever it runs and must never block or allocate. Three slots make that
//! possible: the writer owns one, the reader owns one, and the third ("back") is swapped
//! through a single atomic byte. Writing = fill your slot, then swap it with the back slot
//! (marking it fresh). Reading = if the back slot is fresh, swap your slot with it. Neither side
//! ever waits, and the reader always sees a complete, most-recent value (older ones are simply
//! overwritten: "latest wins", which is what a control signal wants).
use std::{
    cell::UnsafeCell,
    sync::{atomic::{AtomicU8, Ordering}, Arc},
};

const INDEX: u8 = 0b011;
const FRESH: u8 = 0b100;

struct Shared<T> {
    slots: [UnsafeCell<T>; 3],
    /// Index of the back slot (bits 0-1) + FRESH flag (bit 2).
    back: AtomicU8,
}

// Safety: each slot is accessed by exactly one side at a time; ownership moves only through the
// atomic swap (AcqRel orders the slot writes before the hand-off and after the take-over).
unsafe impl<T: Send> Sync for Shared<T> {}

pub struct Writer<T> { shared: Arc<Shared<T>>, idx: u8 }
pub struct Reader<T> { shared: Arc<Shared<T>>, idx: u8 }

/// Create the pair; every slot starts as a clone of `init` (all allocation happens here).
pub fn triple<T: Clone + Send>(init: T) -> (Writer<T>, Reader<T>) {
    let shared = Arc::new(Shared {
        slots: [UnsafeCell::new(init.clone()), UnsafeCell::new(init.clone()), UnsafeCell::new(init)],
        back: AtomicU8::new(1),
    });
    (Writer { shared: shared.clone(), idx: 0 }, Reader { shared, idx: 2 })
}

impl<T> Writer<T> {
    /// Fill the writer's slot in place with `f`, then publish it.
    pub fn write(&mut self, f: impl FnOnce(&mut T)) {
        // Safety: slot `idx` is owned by the writer until the swap below.
        f(unsafe { &mut *self.shared.slots[self.idx as usize].get() });
        let prev = self.shared.back.swap(self.idx | FRESH, Ordering::AcqRel);
        self.idx = prev & INDEX;
    }
}

impl<T> Reader<T> {
    /// The newest published value, or None if nothing new since the last call.
    pub fn read(&mut self) -> Option<&T> {
        if self.shared.back.load(Ordering::Relaxed) & FRESH == 0 { return None; }
        let prev = self.shared.back.swap(self.idx, Ordering::AcqRel);
        self.idx = prev & INDEX;
        // Safety: slot `idx` is now owned by the reader.
        Some(unsafe { &*self.shared.slots[self.idx as usize].get() })
    }
}

unsafe impl<T: Send> Send for Writer<T> {}
unsafe impl<T: Send> Send for Reader<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latest_wins_and_no_tearing() {
        let (mut w, mut r) = triple([0u64; 64]);
        assert!(r.read().is_none());
        w.write(|s| s.fill(1));
        w.write(|s| s.fill(2));
        assert_eq!(r.read().unwrap()[0], 2);
        assert!(r.read().is_none());
        // Concurrent: every slot read must be internally consistent and never go backwards.
        let t = std::thread::spawn(move || {
            for v in 3..200_000u64 { w.write(|s| s.fill(v)); }
        });
        let mut last = 2;
        while last < 199_999 {
            if let Some(s) = r.read() {
                assert!(s.iter().all(|&x| x == s[0]), "torn read");
                assert!(s[0] >= last);
                last = s[0];
            }
        }
        t.join().unwrap();
    }
}
