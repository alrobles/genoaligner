# RESULTADO H12-SW-PARASAIL — throughput SW: GPU vs parasail, Fase D

Fecha: 2026-09-12 · Rama: main · Commit: 5a5700f
Job GPU: **29230691** (MI210, r06r26n01, `gres=gpu:mi210:1`; CPU host: AMD EPYC
9454P 48-core)

## Qué se midió — declarado antes de los números

Tres columnas, intercaladas dentro de cada uno de 7 reps (diseño B6, mismo job,
mismo workload: 600 pares, 105.3 Mcells/rep):

- **gpu-dev**: `sw_score_kernel`, un launch para el batch entero, tiempo de
  device vía hipEvent (más wall para contexto).
- **parasail-1t**: `parasail_sw` (dispatcher al mejor ISA del CPU), UN hilo.
- **parasail-8t**: las mismas llamadas repartidas en 8 `std::thread` — el techo
  que un pipeline CPU real vería, no un feature de parasail.

Cada rep **re-verificó los 600 scores** contra la referencia serial
independiente y contra parasail: **0 mismatches en 7 reps**. La convención de
gaps de parasail se verificó antes de medir (es `open + (L-1)·ext`, igual que la
nuestra — pin `ACGTTTACGT×ACGTACGT → 5`).

## Números medidos

| columna | mean | sd | min | max | spread | throughput |
|---|---|---|---|---|---|---|
| gpu-dev | 32.94 ms | 1.80 | 31.80 | 35.82 | 12.2% | **3.20 GCUPS** |
| gpu-wall | 32.97 ms | 1.80 | 31.82 | 35.85 | 12.2% | 3.19 GCUPS |
| parasail-1t | 156.59 ms | 0.03 | 156.55 | 156.63 | 0.1% | 0.67 GCUPS |
| parasail-8t | 89.99 ms | 1.82 | 88.61 | 93.06 | 4.9% | 1.17 GCUPS |

**gpu-dev vs parasail-1t: 4.8x · gpu-dev vs parasail-8t: 2.7x**

## Lectura honesta

- El kernel SW en MI210 es **~5x más rápido que parasail a un solo core** y
  ~2.7x que parasail a 8 cores en este workload (600 pares ≤1000×1000). Es un
  resultado real pero modesto — no hay que redondearlo hacia arriba.
- El spread de GPU es 12.2% porque los reps 5-6 subieron a ~35.8ms — el nodo
  tenía un co-tenant (`a460m363`, otra GPU física); no hubo nodo MI210
  completamente idle disponible en esta ventana. Se reporta como está.
- `parasail-8t` solo escala 0.67→1.17 GCUPS (1.75x en 8 hilos) — el workload
  es memory-bound del lado CPU; también se reporta como está.
- Comparación válida solo para **score-only**: parasail_sw y nuestro score
  kernel computan la misma cosa. El trace kernel queda fuera de esta medición.
- V100 no medido en este job — la cifra es MI210-específica.
