---
layout: post
title: 	"3rd-advanced section"
date: 	2024-07-26 15:52:17 +0900
categories: KhuDa
---

# 5장 결정트리
## 들어가며
판다스 데이터  
프레임의 각 열
info() 메소드 : 
각 열의 
- 데이터 타입과 
- 누락된 ㅇ데이터가 있는지 확인하는 데 사용한다.
- describe() 메소드
메서드는 열에 대한 간략한 통계를 출력해줍니다. 최소, 최대 , 평균값 등을 볼 수 있다.
mean, std, min, max, mideum, 등

1) 판다스 데이터 프레임을 다룰 때
    - **다수의 열**의 이름이 들어간 경우는
        ```python
        data = wine[['','','']]
        ```
        같이 **이중배열**을 이용하고, 안에 str 이름을 넣는다. 
    - numpy이로 바꾸려면
        - data = wine[['alcohol','sugar','pH']].to_numpy()
    - **하나의 특성**인 target을 이용하는 경우는
        ```python
        data = wine['class'].to_numpy()
        ```
2) train을 다룰 때
    ```python
    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target =train_test_split(data, target, test_size=0.2, random_state=42)
    #20%의 비율로 나눈다는 말이다.
    ```
    ```python
    print(train_input.shape())
    # (5197,3)>> 5197개의 sample에는 3개의 특성있다.
    ```
3) 정규화 
    ```python
    from sklearn.preprocessing import StandardScaler
    ss= StandardScaler()
    ss.fit(train_input)
    train_scaled = ss.transform(train_input)
    test_scaled = ss.transform(test_input)
    # fit과 transform을 둘 다 써야하는 불문율이 있다.
    ```
4) logistic 훈련
    ```python
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(train_scaled, train_target)
    lr.score(train_scaled, train_target)
    lr.score(test_scaled, test_target)
    # 0.78
    # 0.77
    # 절라 낮다. 0.95 이상은 되야 겸상가능
    #이는 과소적합이 되었다.
    # 이에 대한 매개변수를 찾아보기
    ```
page 221
5) 훈련된 결과에 대한 계수 뽑기
    ```python
    lr.intercept_
    lr.coef_
    ```
- 또다른 문제 : 설명력이 **병신**이라고 말할 수 있을 정도로 역하다.

## 결정트리 Desicion tree
- 클래스 DecisionTreeClassifier
- fit()
- score()
- 사용예제
    ```python
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(train_scaled, train_target)
    print(dt.score(train_scaled,train_target))
    print(dt.score(test_scaled,test_target))
    # 0.99
    # 0.85
    # train에 비해 test가 졸라 낮기에 
    # 과대적합이다.
    ```
- 시각화
    ```python
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    plt.figure(figsize=(10,7))
    plot_tree(dt)
    plt.show()
    # Desicion tree가 만들어진다.
    ```
    - 설명
        - 맨 위의 node : root node
        - 맨 아래 끝에 달린 노드 : leaf node
    - 노드란?
        - 훈련 데이터의 특성에 대한 테스트를 표현한다.
    - 함수 설명
        - plot_tree() 함수에서 
            - max_depth
                - 트리의 깊이를 제한하려면?? 
                - max_depth 매개변수를 1로 주면 루트 노드를 제외하고 하나의 노드를 더 확장하여 그린다.
            - filled
                - 색을 넣고 싶다면?
                - filled 매개변수에 역할을 한다.
            - feature_names
                - 특성의 이름을 전달하려면?
                - feature_names 매개변수
    ```python
    plt.figure(figsize=(10,7))
    plot_tree(dt,max_depth=1,filled=True, feature_names=['acohol','sugar','pH'])
    plt.show()
    ```
    # page 226에서 그림 가져오기 

- DesicionTreeClassifier 클래스 설명
    1) criterion 매개변수 - **불순도**
        - default : 'gini'
        - gini 계산하는 방법 : 1-(음성rate^2 + 양성rate^2)
        - 해석 : 불순도가 낮을 수록 0에 가깝다.
        - 작동방식 : 부모 노드와 자식노드간의 불순도 차이가 크도록 트리를 성장시킨다.
            - 계산식
                - 정보이득 (information gain)= 부모의 불순도 - {(1번 자식의 불순도)* (1번 자식 개수/부모)+ ,,,,,}
        - other : 'entropy'
            - entropy 계산하는 방법 :  - (음성 클래스 비율) * log2(음 클 비) - (양성 클래스 비율) * log2(양성 클래스 비율)

    - 문제 : **트리가 제한없이 자라**나기에 train과 test의 차이가 나옴.
    2) max_depth 매개변수 - **가지치기**
        - 가지의 길이가 3개인 경우의 예제 코드
        ```python
        dt = DesicionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(train_scaled, train_target)
        dt.score(train_scaled, train_target)
        dt.score(train_scaled, train_target)
        #0.84
        #0.84
        # 훈련은 낮아지고, 테스트는 거의 그대로이다.
        ```
    - 갑자기 든 생각?
        - 굳이 정규화를 할 필요가 있나?
        - 불순도가 클래스의 비율로 구하기 때문이다.
        ```python
        dt = DesicionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(train_input, train_target)
        dt.score(train_input, train_target)
        dt.score(train_input, train_target)
        #0.84
        #0.84
        ```
    3) feature_importances_ 메소드
    - 각 특성의 중요도를 리스트 형태로 저장한 함수이다.