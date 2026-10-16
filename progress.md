# Withdrawal coverage implementation — complete

- All work is isolated to `Magnetic 1 - Capsule Camera - Withdrawal v1`; prior viewer and solver folders remain unchanged.
- Reverse replay is continuous, maintains forward body orientation, applies slowdown only, and recomputes magnetic dynamics. Duration: 334.13 s; planned waits: 0.
- VTK camera and surface coverage were recomputed on the final retimed trajectory. Added unique area: 16,978.93 mm² (7.64 percentage points of recoverable area).
- Final coverage equals 177,063.66 mm² insertion coverage plus 16,978.93 mm² withdrawal additions. Fold-hidden area is excluded per the approved project rule.
- HTML and EXE were rebuilt; embedded HTML SHA-256 matches the delivered page. Browser smoke test reached both ends of the timeline and switched to/from the original capsule scheme without runtime errors.
- 33 unit tests pass; Python and JavaScript syntax checks pass. All 153 source files in the read-only manifest match their original hashes.
- Safety limitation: minimum centerline clearance is 0 mm; capsule-to-wall clearance and clinical safety are not established.
