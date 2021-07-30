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
python SVR_rbf.py --station 312 
python SVR_rbf.py --station 313 
python SVR_rbf.py --station 314 
python SVR_rbf.py --station 315 
python SVR_rbf.py --station 316 
python SVR_rbf.py --station 371 
python SVR_rbf.py --station 372 
python SVR_rbf.py --station 373 
python SVR_rbf.py --station 374 
python SVR_rbf.py --station 393
python SVR_rbf.py --station 394 
python SVR_rbf.py --station 396 


#运行GBRT
python GBRT.py --station 312 
python GBRT.py --station 313 
python GBRT.py --station 314 
python GBRT.py --station 315 
python GBRT.py --station 316 
python GBRT.py --station 371 
python GBRT.py --station 372 
python GBRT.py --station 373 
python GBRT.py --station 374 
python GBRT.py --station 393 
python GBRT.py --station 394
python GBRT.py --station 396 


# #运行lstm_
python lstm_.py --station 312  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 313  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 314  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 315  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 316  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 371  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 372  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 373  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 374  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 393  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 394  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7
python lstm_.py --station 396  --epochs 100 --batch-size 100  --lr 1e-3 --input-dim 3  --seq-len 7


# #运行mlp
python mlp.py --station 312 --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 313  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 314  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 315  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 316  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 371  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 372  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 373  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 374  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 393  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 394  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7
python mlp.py --station 396  --epochs 100 --batch-size 30  --lr 1e-3 --input-dim 3  --seq-len 7

# # #运行seq2seq
python seq2seq.py --station 312  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 313  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 314  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 315  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 316  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 371  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 372  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 373  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 374  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 393  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 394  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7
python seq2seq.py --station 396  --epochs 100 --batch-size 30  --lr 5e-3 --input-dim 3  --seq-len 7


# # #运行att_seq2seq
python att_seq2seq.py --station 312  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 313  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 314  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 315  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 316  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 371  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 372  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 373  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 374  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 393  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 394  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7
python att_seq2seq.py --station 396  --epochs 120 --batch-size 32  --lr 5e-3 --input-dim 3  --seq-len 7


# # #运行XGB
python XGB.py --station 312 
python XGB.py --station 313 
python XGB.py --station 314 
python XGB.py --station 315 
python XGB.py --station 316 
python XGB.py --station 371 
python XGB.py --station 372 
python XGB.py --station 373 
python XGB.py --station 374 
python XGB.py --station 393 
python XGB.py --station 394 
python XGB.py --station 396 