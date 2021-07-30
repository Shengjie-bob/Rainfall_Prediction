#！/bin/bash
# 第一个bash文件
clear;
echo 'Hello';
# source activate py3 

#运行ARIMA
# python ARIMA.py --station 312 
# python ARIMA.py --station 313 
# python ARIMA.py --station 314 
# python ARIMA.py --station 315 
# python ARIMA.py --station 316 
# python ARIMA.py --station 371 
# python ARIMA.py --station 372 
# python ARIMA.py --station 373 
# python ARIMA.py --station 374 
# python ARIMA.py --station 393 
# python ARIMA.py --station 394 
# python ARIMA.py --station 396 

# #运行SVR_rbf
python SVR_rbf.py --station 312 --ifshuffle
python SVR_rbf.py --station 313 --ifshuffle
python SVR_rbf.py --station 314 --ifshuffle
python SVR_rbf.py --station 315 --ifshuffle
python SVR_rbf.py --station 316 --ifshuffle
python SVR_rbf.py --station 371 --ifshuffle
python SVR_rbf.py --station 372 --ifshuffle
python SVR_rbf.py --station 373 --ifshuffle
python SVR_rbf.py --station 374 --ifshuffle
python SVR_rbf.py --station 393 --ifshuffle
python SVR_rbf.py --station 394 --ifshuffle
python SVR_rbf.py --station 396 --ifshuffle


#运行GBRT
python GBRT.py --station 312 --ifshuffle
python GBRT.py --station 313 --ifshuffle
python GBRT.py --station 314 --ifshuffle
python GBRT.py --station 315 --ifshuffle
python GBRT.py --station 316 --ifshuffle
python GBRT.py --station 371 --ifshuffle
python GBRT.py --station 372 --ifshuffle
python GBRT.py --station 373 --ifshuffle
python GBRT.py --station 374 --ifshuffle
python GBRT.py --station 393 --ifshuffle
python GBRT.py --station 394 --ifshuffle
python GBRT.py --station 396 --ifshuffle


# #运行lstm_
python lstm_.py --station 312 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 313 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 314 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 315 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 316 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 371 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 372 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 373 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 374 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 393 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 394 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7
python lstm_.py --station 396 --ifshuffle --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 1  --seq-len 7


# #运行mlp
python mlp.py --station 312 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 313 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 314 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 315 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 316 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 371 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 372 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 373 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 374 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 393 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 394 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7
python mlp.py --station 396 --ifshuffle --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 1  --seq-len 7

# # #运行seq2seq
python seq2seq.py --station 312 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 313 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 314 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 315 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 316 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 371 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 372 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 373 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 374 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 393 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 394 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7
python seq2seq.py --station 396 --ifshuffle --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 1  --seq-len 7


# # #运行att_seq2seq
python att_seq2seq.py --station 312 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 313 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 314 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 315 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 316 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 371 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 372 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 373 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 374 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 393 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 394 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7
python att_seq2seq.py --station 396 --ifshuffle --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 1  --seq-len 7


# # #运行XGB
# python XGB.py --station 312 --ifshuffle
# python XGB.py --station 313 --ifshuffle
# python XGB.py --station 314 --ifshuffle
# python XGB.py --station 315 --ifshuffle
# python XGB.py --station 316 --ifshuffle
# python XGB.py --station 371 --ifshuffle
# python XGB.py --station 372 --ifshuffle
# python XGB.py --station 373 --ifshuffle
# python XGB.py --station 374 --ifshuffle
# python XGB.py --station 393 --ifshuffle
# python XGB.py --station 394 --ifshuffle
# python XGB.py --station 396 --ifshuffle