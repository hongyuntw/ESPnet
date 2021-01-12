 #!/bin/bash
train_cmd="utils/run.pl"
decode_cmd="utils/run.pl"
train_cmvn="/home/nlp/ASR/espnet/egs/FSW/data/train_nodev/cmvn.ark"
. ./path.sh

# Feature extraction
for x in single_wav ; do
 steps/make_fbank_pitch.sh --nj 1 --write_utt2num_frames true $x exp/make_mfbank/$x single_wav/fbank
 dump.sh --nj 1 --do_delta false single_wav/feats.scp ${train_cmvn} exp/make_mfbank/$x single_wav/fbank/dump
 utils/fix_data_dir.sh --cmd ${decode_cmd} $x
done
