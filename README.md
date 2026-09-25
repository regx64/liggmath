# liggmath

선형대수학(Linear Algebra)과 AI 관련 수학 유틸리티를 제공하는 Python 라이브러리입니다.
NumPy/SciPy를 기반으로, 벡터/행렬 연산부터 분해, 통계, 신경망에 필요한 활성화 함수·손실
함수·옵티마이저·초기화·간단한 자동미분까지 하나의 패키지에서 제공하는 것을 목표로 합니다.

이 저장소는 계속 확장되는 기반(foundation)으로 설계되었습니다 — 새로운 수학 유틸리티는
기존 서브패키지에 함수를 추가하거나 새 서브패키지를 만드는 방식으로 자연스럽게 확장됩니다.

## 설치

```bash
pip install -e .
# 또는 개발/테스트 의존성까지
pip install -e ".[dev]"
```

의존성: `numpy`, `scipy`

## 구조

```
liggmath/
├── linalg/       # 선형대수: 벡터, 행렬, 분해, 연립방정식
├── stats/        # 통계: 기술통계, 확률분포
├── ai/           # AI 수학: 활성화함수, 손실함수, 평가지표, 초기화, 자동미분, 옵티마이저
└── utils/        # 수치 유틸리티: 그래디언트 체크, 데이터 전처리
```

## 빠른 시작

```python
import numpy as np
from liggmath import linalg, stats, ai, utils

# --- 선형대수 ---
a = np.array([3.0, 4.0])
linalg.norm(a)                      # 5.0
linalg.normalize(a)                 # 단위벡터

m = np.array([[4.0, 3.0], [6.0, 3.0]])
Q, R = linalg.qr(m)                 # QR 분해
U, S, Vt = linalg.svd(m)            # 특이값 분해
x = linalg.solve(m, np.array([1.0, 2.0]))  # 연립방정식 풀이

# --- 통계 ---
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
stats.mean(data), stats.std(data)
stats.standardize(data)             # z-score 정규화

# --- AI 수학 ---
x = np.array([-1.0, 0.0, 1.0, 2.0])
ai.relu(x)
ai.softmax(x)
ai.mse(y_true=np.array([1.0, 0.0]), y_pred=np.array([0.9, 0.1]))
w = ai.he_normal(fan_in=128, fan_out=64, seed=0)   # 가중치 초기화

# 스칼라 자동미분 (교육/프로토타이핑용)
a = ai.Value(2.0); b = ai.Value(-3.0); c = ai.Value(10.0)
out = (a * b + c).relu()
out.backward()
a.grad, b.grad, c.grad

# 옵티마이저
params = [np.array([5.0])]
opt = ai.Adam(params, lr=0.1)
opt.step(grads=[2 * params[0]])

# --- 유틸 ---
utils.one_hot(np.array([0, 1, 2]), num_classes=3)
utils.numerical_gradient(lambda v: float(np.sum(v ** 2)), np.array([1.0, 2.0]))
```

## 모듈별 제공 기능

### `liggmath.linalg`
- **벡터**: `dot`, `norm`, `normalize`, `angle_between`, `cross`, `project`, `is_orthogonal_vectors`
- **행렬**: `identity`, `zeros`, `ones`, `random_matrix`, `transpose`, `trace`, `rank`,
  `determinant`, `inverse`, `pseudo_inverse`, `is_symmetric`, `is_orthogonal_matrix`,
  `is_positive_definite`, `matmul`, `hadamard`, `kronecker`
- **분해**: `lu`, `qr`, `svd`, `eig`, `eigh`, `cholesky`
- **연립방정식**: `solve`, `least_squares`, `solve_triangular`

### `liggmath.stats`
- **기술통계**: `mean`, `variance`, `std`, `covariance`, `covariance_matrix`,
  `correlation_matrix`, `standardize`, `moving_average`
- **확률분포**: `normal_pdf`, `normal_cdf`, `binomial_pmf`, `poisson_pmf`

### `liggmath.ai`
- **활성화 함수 (+도함수)**: `sigmoid`, `relu`, `leaky_relu`, `tanh`, `softmax`, `gelu`,
  `silu`, `elu`
- **손실 함수**: `mse`, `mae`, `rmse`, `binary_cross_entropy`, `categorical_cross_entropy`,
  `huber_loss`
- **평가지표**: `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `r2_score`
- **가중치 초기화**: `xavier_uniform`, `xavier_normal`, `he_uniform`, `he_normal`
- **자동미분**: `Value` (스칼라 reverse-mode autograd, micrograd 스타일)
- **옵티마이저**: `SGD`, `Momentum`, `Adam`

### `liggmath.utils`
- `numerical_gradient` (그래디언트 체크용 수치미분), `clip_gradients`, `one_hot`,
  `shuffle_data`, `train_test_split`, `set_seed`

## 테스트

```bash
pytest
```

## 로드맵

이 라이브러리는 계속 확장될 예정입니다. 예를 들어:
- 텐서/N차원 연산, FFT, 컨볼루션 유틸
- 더 다양한 확률분포와 샘플링
- 배치 자동미분(텐서 기반 autograd)
- 최적화 알고리즘(경사하강 변형, 뉴턴법 등) 확대
- 신호처리, 그래프 이론 유틸

기여 및 제안은 이슈/PR로 환영합니다.
