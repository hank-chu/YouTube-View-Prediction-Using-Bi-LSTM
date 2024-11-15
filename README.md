# YouTube-View-Prediction-Using-Bi-LSTM
機器學習：利用Bi-LSTM訓練模型從YouTube影片標題預測點閱數

Machine Learning: Predicting Views from YouTube Video Titles Using Bi-LSTM Training Models

此專案使用自然語言處理 (NLP) 和深度學習模型 Bi-LSTM，透過分析 YouTube 影片的標題、發佈時間和頻道訂閱數，來預測影片的點閱率。該專案能夠幫助內容創作者了解哪些影片可能會受到歡迎，也為影片流量預測提供了一種有效的解決方案。

## 功能特點
自動化數據收集：使用 YouTube API 抓取熱門影片資訊，包括標題、發布時間、觀看次數和頻道訂閱數。

自然語言處理：使用 ckiptagger 進行斷詞，並將標題轉換為向量表示，以便後續分析。

深度學習預測：使用 Bi-LSTM 模型對影片的標題向量、發佈時間和訂閱數進行學習，預測影片可能的觀看次數。

視覺化：利用 t-SNE 和 PCA 進行詞向量的可視化，並生成 2D 與 3D 散點圖，直觀展示詞語分佈。

高準確度：多個模型測試中，最佳模型在標題、時間和訂閱數的綜合因素下，能達到良好的預測準確度。

## 關鍵字 / Keywords
- YouTube Data Analysis

- NLP

- Bi-LSTM

- Word2Vec / Doc2Vec

## 專案結構
- Youtube_Crawler.py：從指定頻道抓取影片資訊，包括影片 ID、標題、發布時間、觀看次數等，並儲存至 CSV 檔案。
- _get_top50_YouTube.py：透過 YouTube API 抓取台灣地區熱門影片的頻道資訊。
- github_getdata.py：資料前處理函數，包括斷詞、簡繁轉換等操作。
- model_F.py：定義數據處理和模型訓練流程，並保存模型以供後續預測使用。
- _Word2Vec_visualization_3D.py：將 Word2Vec 結果以 3D 散點圖顯示，並匯出動畫 GIF。
- _Doc2Vec_visualization.py：使用 Doc2Vec 技術將影片標題向量化，並透過 t-SNE 進行可視化。
- github_predict.py：使用訓練好的 Bi-LSTM 模型進行點閱率預測。


## 使用方法
### 1. 抓取熱門頻道資料
使用 Youtube_Crawler.py 抓取台灣地區熱門影片的頻道資訊：

```bash
python Youtube_Crawler.py
```

### 2. 訓練並保存 LSTM 模型
運行 model_F.py 進行 Bi-LSTM 模型訓練，並將模型保存為 LSTM_model.h5：

```bash
python model_F.py
```
![image](https://github.com/user-attachments/assets/9a76ba0d-6b4c-4040-9a49-f478ce6934f3)

![image](https://github.com/user-attachments/assets/be25a08b-48d6-4ef1-83ca-1ad6567fb052)


### 3. 可視化詞向量
透過 _Doc2Vec_visualization.py 或 _Word2Vec_visualization_3D.py 進行詞向量的 2D 或 3D 視覺化：

```bash
python _Word2Vec_visualization_3D.py
```
![_Word2Vec_visualization_3D](https://github.com/user-attachments/assets/ec44359f-48dd-45e3-9bca-dfe913b62331)


```bash
python _Doc2Vec_visualization.py
```
![_Doc2Vec_visualization](https://github.com/user-attachments/assets/97c9521b-743b-4759-a347-b0249448085b)


### 4. 預測 YouTube 點閱率
在 github_predict.py 中輸入頻道 ID 和影片 ID，使用訓練好的模型進行點閱率預測：

```bash
python github_predict.py
```
![預測](https://github.com/user-attachments/assets/7749be08-698e-4416-a3f9-e87a44396f7a)



## 結果分析
該模型在影片標題、發布時間和訂閱數的綜合因素下，能夠有效預測影片的觀看數。最佳模型顯示了標題文本向量、時間因素和頻道訂閱數對預測準確度的重要性。


