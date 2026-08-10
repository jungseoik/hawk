# References (참고 문헌)

본 원고는 원본 HAWK 논문의 참고문헌 번호 **[1]–[49]** 를 그대로 승계하고
(`../origin/07_references.md`), CERBERUS에서 새로 인용하는 문헌에 **[50]** 부터 번호를
이어 붙인다. 최종 투고 시 학회 양식에 맞춰 전체를 재정렬하되, 그 전까지는 이 승계 규칙을
유지해 본문의 번호가 흔들리지 않게 한다.

> **작성 상태.** [1]–[49]는 원본 목록을 그대로 인용하므로 여기 재수록하지 않는다.
> 아래는 본 연구가 추가한 항목이며, 투고본에서는 [1]–[49]와 병합하여 단일 목록으로 낸다.

---

## 본 연구가 추가한 문헌

[50] Tang, J., Lu, H., Wu, R., Xu, X., Ma, K., Fang, C., Guo, B., Lu, J., Chen, Q.,
Chen, Y.-C.: Hawk: Learning to Understand Open-World Video Anomalies.
In: Advances in Neural Information Processing Systems (NeurIPS) (2024)

> 본 연구가 **직접 확장하는 베이스 연구**다. 아키텍처(외형+동적 이중 브랜치, 병목 사영),
> 학습 인프라, Stage-2 이상행동 데이터셋을 물려받는다. 상속분과 본 연구의 델타는
> Section 2.1·3.3·3.4에 명시되어 있으며, Table 1에는 원 논문이 공개한 수치를 별도 행으로
> 싣는다(본 연구가 재학습한 Dynamic-only 구성과 동일 대상이 아니므로 함께 보고한다).

[51] Simonyan, K., Zisserman, A.: Two-Stream Convolutional Networks for Action
Recognition in Videos. In: Advances in Neural Information Processing Systems
(NeurIPS) (2014)

[52] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M.,
Mohamed, S., Lerchner, A.: beta-VAE: Learning Basic Visual Concepts with a
Constrained Variational Framework. In: International Conference on Learning
Representations (ICLR) (2017)

[53] Tishby, N., Pereira, F.C., Bialek, W.: The Information Bottleneck Method.
In: Proceedings of the 37th Annual Allerton Conference on Communication,
Control and Computing, pp. 368–377 (1999)

[54] Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S.: Barlow Twins:
Self-Supervised Learning via Redundancy Reduction. In: International Conference
on Machine Learning (ICML) (2021)

[55] Bardes, A., Ponce, J., LeCun, Y.: VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning. In: International Conference on
Learning Representations (ICLR) (2022)

---

## 검증 필요 (투고 전 필수)

아래 항목은 기억이 아니라 **원문을 직접 확인해 서지 정보를 검증**한 뒤 확정해야 한다.
저자명 철자, 게재 연도, 학회명·권호·페이지가 모두 대조 대상이다.

- [51]–[55]: 위 정보는 통상적으로 알려진 서지사항을 기재한 것으로, 원문 대조 전이다.
- [50]: `README.md`의 BibTeX 항목에서 옮겼으며, 원 논문 PDF
  (`../origin/hawk_neurips2024.pdf`)와 대조 완료.
- 본문에서 번호로 인용하나 원본 목록에 없는 문헌이 더 있는지 전수 점검할 것
  (`grep -o "\[[0-9]\+\]" improved/*.md` 로 사용 번호를 모아 [1]–[55]와 대조).
