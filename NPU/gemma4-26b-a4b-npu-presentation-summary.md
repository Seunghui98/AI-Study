# Gemma4-26B-A4B(MoE)를 NPU에서 돌리기

nntrainer PR #4428 / device : Galaxy S26 Ultra (Hexagon v81), 일부 S25 Ultra

## 한 줄 요약

Gemma4-26B-A4B의 prefill과 decode를 모두 NPU에서 돌리게 만들었고,
prefill은 약 400 → 700 TPS로 1.7배 빨라졌고 peak 메모리는 4 GB 안팎에 맞췄습니다.

| | 처음 (10/10) | 지금 |
|---|---|---|
| prefill 1024토큰 | 402 TPS | 627 TPS |
| prefill 2048토큰 | 398 TPS | 약 700 TPS |
| 첫 토큰 준비 시간 | 1,463 ms | 506 ms |
| decode | 10~13 TPS | 13~15 TPS (2-bit / ternary pack), 19.5 TPS (최신 pack) |
| peak 메모리 | - | 3.6~4.2 GB |

---

## 1. 지난 발표 이후 진행 상황

지난번 계획으로 잡았던 항목은 모두 끝냈습니다.

- Gemma4-26B-A4B를 NPU에서 돌리기
- MoE expert, projection, RMSNorm, RoPE, router, lm_head, attention을 NPU로 옮기기
- decode까지 NPU에서 돌리기

지금은 tokenizer, embedding 조회 같은 일부만 CPU에 남아 있고, 앱(SampleTestAPP)에서도 NPU 백엔드로 선택해 쓸 수 있습니다.

---

## 2. 왜 어려웠나

- **모델이 NPU 메모리보다 큽니다.** weight가 6.6~7.5 GB인데, NPU가 한 번에 매핑할 수 있는 건 3.8 GB입니다. 그래서 expert를 layer당 16개만 올려 두고 나머지는 필요할 때 flash에서 읽습니다.
- **expert가 LFM2보다 4배 많습니다.** layer당 128개 중 8개를 고르고, layer가 30개입니다.
- **NPU로 옮긴 연산이 늘면서 호출 비용이 커졌습니다.** 층마다 NPU를 따로 부르면 matmul보다 호출·복사 시간이 더 커집니다.

---

## 3. Prefill : "부르는 횟수와 복사를 줄이고, 계산 유닛을 쉬지 않게"

### 호출 횟수 줄이기

작은 연산들을 큰 호출 하나로 합쳤습니다.
q/k/v를 한 호출로, dense FFN 3개를 한 호출로, norm과 residual add는 앞 matmul 호출 안으로 넣었습니다.
그 결과 layer당 NPU 호출이 7번에서 2번으로 줄었습니다.

행이 많아 한 번에 안 들어갈 때도 CPU가 여러 번 나눠 부르지 않고, NPU 안에서 알아서 나눠 돌게 했습니다.
NPU 호출은 버퍼 크기에 비례하는 고정 비용이 있어서, 나눠 부를수록 그 비용을 또 내기 때문입니다.

→ S25U 1024토큰 기준 4.8초 → 3.8초

### 복사 줄이기

예전에는 NPU 호출이 끝날 때마다 결과를 CPU로 가져왔다가 다음 호출에 다시 넣었습니다.
이제 한 블록 안의 중간 결과는 NPU 쪽 버퍼에 그대로 둡니다.

→ 2048토큰 prefill의 복사 시간 402 ms → 39 ms

### MoE 계산 다듬기

- 한 토큰이 expert 8개로 가면 같은 값을 8번 양자화하던 것을 1번만 하게 했습니다. (2048토큰 약 4.6초 → 4.0초)
- 빈 칸(padding)만 있는 부분은 계산을 건너뜁니다. (2048토큰 3.2초 → 3.0초)
- 행렬 유닛(HMX)이 다음 계산을 하는 동안, 벡터 유닛(HVX)이 앞 결과를 후처리하게 겹쳤습니다.

### Attention

- 점수 계산부터 softmax까지 NPU 내부 메모리(VTCM) 밖으로 나가지 않는 int8 attention kernel을 새로 만들었습니다.
- 긴 문맥(4,096토큰 이상)에서 느린 fp16 경로로 떨어지던 문제를 없앴습니다. (2.8만 토큰 prefill 69.5초 → 39.9초)

### prefill 결과 (S26U)

| | 1024토큰 | 2048토큰 |
|---|---|---|
| 처음 (10/10) | 2.55초 (402 TPS) | 398 TPS |
| 지금 | 1.63초 (627 TPS) | 2.92초 (700 TPS) |

---

## 4. Decode : "토큰 하나를 NPU 안에서 끝내기"

### 구조를 바꿨습니다

decode 한 step 전체를 NPU 안에 그래프로 올려 두었습니다.
CPU는 "다음 위치"만 알려 주고, NPU는 계산을 끝낸 뒤 다음 토큰 번호만 돌려줍니다.
토큰 하나에 메시지 한 번만 오가고, 그동안 CPU는 쉽니다.

### 첫 토큰을 빠르게

첫 토큰이 나오기 전에 prefill KV를 decode용 형태로 옮기고 그래프를 초기화하는 준비 작업이 있어서, 첫 토큰까지 1.5초 가까이 걸렸습니다.
이 작업을 병렬화·벡터화하고, 층마다 30번 보내던 RoPE 테이블을 한 번만 보내게 해서
1,463 ms → 506 ms로 줄였습니다.

### Attention을 가볍게

decode용 KV cache를 int8로 바꾸고 정수 연산 attention을 넣어서
토큰당 attention 시간을 20.9 ms → 약 6.7 ms로 줄였습니다. (S25U → S26U에 걸친 수치)

### decode 속도는 결국 "expert를 몇 개 올려 둘 수 있느냐"

decode 시간의 대부분은 expert weight를 읽는 시간입니다.
그래서 계산을 빠르게 하는 것보다, 필요한 expert가 이미 올라와 있게 하는 게 더 효과가 컸습니다.

- 첫 토큰 이후 남는 메모리만큼 상주 expert를 layer당 16개 → 최대 34개로 늘렸습니다. (10.4 → 13.1 TPS)
- 자주 쓰는 expert가 오래 남도록 교체 정책을 바꿨습니다. (LRU → LRFU)
- 다음 layer에서 쓸 expert를 미리 예측해서 읽어 둡니다. 다만 단독 효과는 크지 않았고, 보조 역할입니다.
- weight 자체를 2-bit → ternary(약 1.67 bit)로 줄였더니 읽을 양이 줄어서 11.8 → 15.5 TPS가 됐습니다.

---

## 5. 메모리

NPU 내부 메모리(VTCM)는 8 MB밖에 없어서, 호출마다 용도에 맞게 나눠 씁니다.

```
VTCM 8 MiB (prefill MoE)          VTCM 8 MiB (decode MoE)
├─ 입력 64행        176 KiB       ├─ expert 버퍼 A   968 KiB  ← 짝수 expert
├─ gate_up weight   1.89 MiB      ├─ expert 버퍼 B   968 KiB  ← 홀수 expert
├─ down weight      0.95 MiB      │   (하나 계산하는 동안 다른 쪽에 다음 expert를 미리 올림)
├─ 중간/결과 버퍼    약 1.8 MiB    └─ 나머지 여유
└─ 합계 약 4.8 MiB
```

expert 1개 크기는 2-bit 기준 1.45 MiB, ternary 기준 1.23 MiB입니다. (LFM2 expert는 5.25 MiB)

전체 메모리에서 크게 줄인 것들입니다.

| 항목 | 전 | 후 |
|---|---|---|
| 3만 토큰 문맥의 KV cache | 6.5 GB | 495 MB |
| sliding attention 층의 decode cache | 640 MiB | 106 MiB |
| expert를 전부 ternary로 | peak 4.7 GB | 3.7 GB |
| lm_head를 NPU로 옮겨 CPU 사본 제거 | 396 MiB | 0 |

---

## 6. GPU와 비교 (같은 칩, 참고용)

| | prefill 1K | decode | peak |
|---|---|---|---|
| GPU (Adreno 840) | 388 TPS | 20~25 TPS | 3.4~7.8 GB |
| NPU | 627 TPS | 13~15 TPS (최신 pack 19.5) | 3.6~4.2 GB |

prefill과 메모리는 NPU가 유리하고, decode는 아직 GPU가 더 빠르거나 비슷합니다.
(GPU 쪽은 weight 포맷이 달라서 정확히 같은 조건의 비교는 아닙니다.)

---

## 7. 남은 과제

- decode 16 TPS 목표 : 최신 pack은 넘었지만 기존 pack은 아직 미달
- 긴 문맥 decode : 3만 토큰에서는 attention이 토큰 시간의 절반을 차지
- 정해진 양자화 범위를 쓰는 static 모드가 아직 느림 (prefill 82 TPS)
- 에이전트 시나리오(PetCare)에서 tool 호출이 기대와 다르게 나옴
