./utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
./utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
./utils/fix_data_dir.sh ./data/train
./utils/fix_data_dir.sh ./data/test
