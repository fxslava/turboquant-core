# Review 3: generated ISA and simulator admission

Strict build passed. Generated scalar WAIT_FLAG route counts and total branch
count match baseline; scalar GM LD.b32 static sites decrease18→17 via shared
prologue acquisition. First mapping load is before the owner rendezvous.
No additional vector LDK/STK spill or device logging exists. Scalar frame address
extent and all extra fixed-frame LD_IO/ST_IO instructions are accounted in
review3.json; a timing gain is not assumed and dynamic CCU/DCache checking is
mandatory. Cross-engine object identity proves Slot03 AIC==baseline and Slot05
AIC==Slot04/AIV==Slot03. Actual data arithmetic remains in shared source.
