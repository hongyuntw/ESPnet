./utils/utt2spk_to_spk2utt.pl data/pilot_test/utt2spk > data/pilot_test/spk2utt
./utils/fix_data_dir.sh ./data/pilot_test
./run.sh --stage 1 --stop-stage 2 
