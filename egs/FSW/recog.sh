./utils/recog_wav.sh --backend "pytorch" \
--decode_dir "/home/nlp/Demo/api/asr_result" \
--cmvn "/home/nlp/ASR/espnet/egs/aishell/asr1/data/train_nodev/cmvn.ark" \
--lang_model "/home/nlp/ASR/espnet/egs/GrandChallenge/exp/train_rnnlm_pytorch_aishell_lm/rnnlm.model.best" \
--recog_model "/home/nlp/ASR/espnet/egs/GrandChallenge/exp/train_pytorch_conformer_self_mix_train/results/model.loss.best" \
--decode_config "/home/nlp/ASR/espnet/egs/GrandChallenge/conf/conformer_self_mix_decode.yaml" \
--api "v1" "/home/nlp/ASR/espnet/egs/FSW/single_wav/inference.wav"

