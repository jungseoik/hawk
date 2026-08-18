---
name: reference-verified-bibliography
description: 온라인 검증 완료된 [50]-[57] 서지정보·BibTeX·출처 URL과 인용 주장 크로스체크 결과(재검증 불필요)
metadata:
  type: reference
---

검증일 **2026-08-11**. `improved/07_references.md`의 "검증 필요" 표기는 이 파일로 해소된다.
8건 모두 **저자·연도·venue 오류 없음**. 남은 문제는 diacritic·middle initial·권/페이지 누락뿐.

| # | 판정 | 고칠 것 |
|---|---|---|
| [50] HAWK | 확인 | proceedings 표기는 `HAWK`(전대문자). arXiv:2405.16886 |
| [51] Two-Stream | 확인 | **venue는 NIPS 2014** (NeurIPS는 2018년 개칭). vol 27, pp. 568–576 |
| [52] beta-VAE | 확인 | **Loïc** Matthey, Christopher **P.** Burgess, Matthew **M.** Botvinick |
| [53] IB | 확인 | pp. 368–377 (Allerton). arXiv:physics/0004057에 journal-ref 없음 → 2차 출처 기반 |
| [54] Barlow Twins | 확인 | PMLR 139, pp. 12310–12320. arXiv:2103.03230 |
| [55] VICReg | 확인 | arXiv:2105.04906 |
| [56] Choi et al. | 확인 | NeurIPS 32, pp. 851–863. arXiv:1912.05534 |
| [57] RESOUND | 확인 | Springer LNCS 11210, pp. **520–535**, DOI 10.1007/978-3-030-01231-1_32 (CVF본은 513–528 — Springer 채택 권장) |

출처: proceedings.neurips.cc / papers.nips.cc / proceedings.mlr.press/v139/zbontar21a.html /
arxiv.org/abs/{2405.16886, 1406.2199, 2103.03230, 2105.04906, 1912.05534, physics/0004057} /
dblp.org/rec/conf/{iclr/HigginsMPBGBML17, eccv/LiLV18, nips/ChoiGMH19} /
link.springer.com/chapter/10.1007/978-3-030-01231-1_32 / openreview.net/forum?id=vBKoEZ1PG3

## 인용 주장 크로스체크 — 정정 3건

1. **Barlow Twins/VICReg 혼동 (원고 §2.3, Appendix A·C).** Barlow Twins는 *두 브랜치 간
   정규화 교차상관 행렬*의 비대각을 0으로(대각은 1로) 민다. VICReg는 교차상관이 아니라
   **각 브랜치의 공분산 행렬** 비대각을 개별적으로 벌하고 분산 하한 항을 더한다(VICReg §5.3에
   그 차이가 명시됨). 원고가 두 논문을 "교차상관 비대각→0"으로 합쳐 서술한 것은 부정확.
2. **[56]/[57] 역할 분담 (원고 §2.4).** [56]=**손실 측** 억제(gradient reversal + Places365
   pseudo scene label 적대 손실, 사람 마스킹 영상에 엔트로피 최대화). [57]=**데이터셋 측**
   구성(representation bias 정량화 후 explicit 리샘플링 / implicit=Diving48 제작).
   "배경만으로 예측 **불가능**하게"는 RESOUND엔 과장(bias 최소화·보정이지 불가능 증명 아님).
3. **HAWK의 모션 브랜치는 옵티컬 플로우 이미지가 아니다.** Farnebäck 플로우의 **크기를
   [0,1]로 정규화한 soft Mask를 원본 RGB에 곱한다**(HAWK 수식 4; 저장소 번역
   `origin/04_methodology.md:45`에도 동일). 즉 HAWK의 모션 스트림도 **원본 픽셀 공간**에
   거주한다. `L_MV = 1 − cos`는 압축("tight") 표현 사이에서 계산되며, 의존 구문 분석은
   별개의 **모션-언어 매칭 손실**(토큰 CE)을 먹인다 — 두 손실을 섞어 쓰면 오귀속.
   HAWK 논문은 파서 이름을 명시하지 않음(spaCy 등 특정 도구 귀속 금지).
   HAWK에 배경/정적 스트림은 없음(확인) — CERBERUS의 제3 브랜치는 실제 공백.

> **파생 결론(중요).** CERBERUS의 차별점은 "플로우 이미지·RGB 차분과 다른 신호 공간"이
> 아니다 — 그 대비는 자기 베이스라인(HAWK)에 대해 성립하지 않는다. 진짜 델타는
> (i) soft mask → **이진 마스크**(화소값이 정확히 한 스트림에만 온전히 배분됨)와
> (ii) **보수 스트림의 인스턴스화**(전용 인코더+장면 언어 감독)뿐이다. 상세는
> [[feedback-cerberus-weaknesses]] R2-1 참조.

BibTeX 전문은 필요 시 이 표의 필드로 재구성 가능. 핵심 수정 3건:
`booktitle={... (NIPS)}` for [51], diacritic 복원 for [52]/[54], Springer 페이지 for [57].
