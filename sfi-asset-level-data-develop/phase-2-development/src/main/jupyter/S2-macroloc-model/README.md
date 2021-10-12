# S2 Macro-localization Model Build for EAINB

These steps create Sentinel-2 RGB chips for using in a macro-localization model. To build the model, run these notebooks in this order. ___Note: waiting on production release of 1.5.0-RC2 to make this generally usuable. Can make use of the chips created at the end by pulling from S3 now. Contact Kim if you want modifications to the chip specifications in the meantime.___

1. 01-S2-RGB-cement-chip-creation.ipynb
2. 02-S2-RGB-cement-chip-upload.ipynb
3. 03-S2-RGB-steel-chip-creation.ipynb
4. 04-S2-RGB-steel-chip-upload.ipynb