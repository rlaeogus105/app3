import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title('데이터 수집 및 분석 애플리케이션')

# 데이터 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기")
    st.write(data.head())
    
    # 데이터 기초 통계량
    st.write("기초 통계량")
    st.write(data.describe())
    
    # 결측값 확인
    st.write("결측값 확인")
    st.write(data.isnull().sum())
    
    # 데이터 시각화
    st.write("데이터 시각화")
    st.write("히스토그램")
    fig, ax = plt.subplots()
    data.hist(ax=ax, figsize=(10, 6))
    st.pyplot(fig)
    
    # 상관관계 히트맵
    st.write("상관관계 히트맵")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, ax=ax)
    st.pyplot(fig)
    
    # 타겟 열 선택
    target_column = st.selectbox("타겟 열을 선택하세요", data.columns)
    
    # 타겟 열을 제외한 피처 선택
    feature_columns = st.multiselect("피처 열을 선택하세요", data.columns.drop(target_column))
    
    if st.button("모델 학습 및 평가"):
        # 데이터 분할
        X = data[feature_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 모델 학습
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 결과 출력
        st.write(f"정확도: {accuracy_score(y_test, y_pred)}")
        st.text("분류 보고서:")
        st.text(classification_report(y_test, y_pred))
        
        # 중요 피처 시각화
        st.write("중요 피처")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        feature_importances.plot(kind='bar', ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)
