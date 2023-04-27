# test on cpu device
python -m torch.distributed.launch tools/main.py -c configs/svtr_seq_ctc.yml --local_rank 0

