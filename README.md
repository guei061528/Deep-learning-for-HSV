# Deep_HSV
Deep learning for writer-dependent handwrite signature verification

The database we use is ICDAR2011.
http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)


1. Pre-processing the original data before training the network. (s_get_normalpic.py)
2. Give the data to the Siamese network to build a personal model for everyone. (s_train_Siamese_com2011.py)
3. Create test data. (Create_TestWDData_com2011.py)
4. Verify test data and establish ROC curve. (s_verify_Siamese_load_com2011.py)
