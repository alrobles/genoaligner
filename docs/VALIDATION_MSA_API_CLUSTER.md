# Validación en clúster — public MSA API (`devin/msa-public-api`)

Estado: el CPU gate prueba la entrada device **bajo el shim** (kernels shipped,
bit-exactitud del código). Lo que el shim no prueba: ejecución en device real,
el build cmake completo con hipcc/nvcc, y rendimiento. Este documento es el
plan para cerrar ese gap y, con él, el lado C++ de la superficie MSA.

## Qué queda por validar (y por qué el shim no basta)

| Brecha | Por qué importa |
|--------|-----------------|
| `genoaligner_msa_api_device_test` en device real | El shim ejecuta los kernels en host; un fallo de empaquetado/arena en device real solo aparece corriendo |
| Build cmake + hipcc/nvcc | Los nuevos targets (`genomsa_gpu`, link PUBLIC a `genoaligner`, `install`, `-x cu` en CUDA) nunca se han compilado con toolchain real |
| `ctest` en nodo GPU | `msa_device_runs` corre de verdad (sin skip); además los tests pairwise corren sobre device real |
| Stage 6 del gate (real sequences) | En hosts sin GPU se salta; en nodo GPU corre `test_real_sequences`, concurrencia y batch contra device |
| Paridad host/device en datos reales | El shim prueba el *código*; la paridad en FASTAs reales de genes_qc_pass es la evidencia end-to-end |
| Timing host vs device | El número de speedup que el paper necesita (ND1/COI completos) |
| Segunda arquitectura (CUDA) | "Portable multiarquitectura" lo exige: mismo .hip por nvcc+nvidia_detail |

## Convenciones heredadas (de msa_gpu_validate*.sbatch)

- Clone congelado por propósito: `/beegfs/a474r867/genoaligner/repo-msa-api`
  @ `devin/msa-public-api` (head al momento de lanzar; último run @ `ff21aef`).
- Partición `sixhour`, logs a `/beegfs/a474r867/genoaligner/logs/`.
- Paridad = **byte-identical**, nunca "close enough".
- FASTAs reales: `/beegfs/a474r867/phylogenyAI/data/genes_qc_pass/`
  (CDS reales — ideales para el path codón; gc=2 en CYTB/COI/ND1/ND2).

## V0 — preparación (desde aquí / login node)

```bash
git -C /beegfs/a474r867/genoaligner/repo-msa-api status 2>/dev/null ||
  git clone -b devin/msa-public-api \
      https://github.com/alrobles/genoaligner-devel \
      /beegfs/a474r867/genoaligner/repo-msa-api
```

## V1 — MI210 (ROCm, arquitectura primaria)

`sbatch scripts/msa_api_gpu_validate.sbatch` — un job, ~15–40 min:

1. `ml load rocm/6.4.3` + `scripts/build_rocm.sh` — build cmake completo
   (libgenoaligner + genomsa_gpu + tests + example). Un fallo aquí significa
   que el wiring de targets/install está roto en toolchain real.
2. `ctest --test-dir build/rocm --output-on-failure` — `msa_api_runs` y
   `msa_device_runs` corren de verdad; los tests pairwise también corren
   sobre device real (bonus: cierra su propia brecha de validación).
3. `genoaligner_msa_api_device_test` directo — PASS requerido, no 77.
4. **Paridad pública end-to-end**: `genoaligner_example_msa` host vs
   `--device` sobre un subset real ($GENE, SUBN=150) — el cuerpo del FASTA
   alineado debe ser byte-idéntico (la línea de resumen difiere por diseño:
   reporta `engine=host|device`).
5. **Path codón**: `--codon --gc 2` sobre CYTB — mismas exigencias.
6. **Timing**: `time` host vs device sobre $BIG (ND1 default) — captura el
   speedup para el paper.
7. `HIPCC=hipcc bash scripts/check_kernel_cpu.sh` — el gate completo en el
   nodo: su stage 6 (real sequences) corre por primera vez con device.

Veredicto: `MSA-API-VAL: PASS` al final del log; cualquier sección roja
aborta con `FAIL`.

## V2 — NVIDIA (portabilidad multiarquitectura)

`sbatch scripts/msa_api_cuda_validate.sbatch` — V100 (sm_70) por defecto;
`--export=ARCH=80,GRES=gpu:a100:1` para A100.

Mismos pasos 1–6 con `GENOALIGNER_ARCH=<sm> scripts/build_cuda.sh`.
Demuestra la promesa del mismo-.hip-fuente: nvcc + `nvidia_detail` produce
el binario device que sirve `msa_align_device`, idéntico al host.

## V3 — cierre

- Parsear logs → tabla PASS/FAIL por sección.
- Pegar la tabla como comentario en PR #12 y marcar el checkbox
  "Real GPU build + run" del test plan.
- Merge PR #12 → `devin/codon-msa`; la siguiente unión a `main` del repo
  de producción lleva la superficie MSA completa.
- Con el lado C++ cerrado → Fase 2 (bindings Rcpp `msa_align`/`msa_codon`).

## Resultados (2026-09-25)

| Leg | Job | Device | Arch | Resultado |
|-----|-----|--------|------|-----------|
| CUDA public API | 30360709 | Quadro RTX 6000 | sm_75 | **PASS** — ctest 9/9, `msa_api_device_test` PASS, paridad DNA+CODON byte-idéntica (34s) |
| CUDA V100 | 30402545 | Tesla V100 | sm_70 | **PASS** — ctest 9/9, device test PASS, paridad DNA+CODON byte-idéntica (42s) |
| ROCm MI210 | 30252647, 30360674, 30360718, 30402538 | MI210 | gfx90a | PENDING — nodos saturados (MIXED+PLANNED; 4 nodos bajo reserva hpc_wang) |

### Hallazgos de esta ronda

- **cmake GLIBCXX (ambos backends)**: `cmake/3.30.3/gcc/14.2` (el primero
  que prueba `build_*.sh`) exige `GLIBCXX_3.4.32`; los nodos traen ≤3.4.29.
  Fix sin tocar scripts: enviar el job con
  `--export=ALL,PATH=/kuhpc/sw/cmake/3.30.3/gcc/11.4/bin:$PATH`
  (la variante gcc/11.4 corre con el libstdc++ del sistema).
- **PIE en nvcc (fixed `ff21aef`)**: los targets pure-C++ no pasaban por
  `genoaligner_configure_hip` → objetos sin `-fPIE` rompían el link PIE
  (`R_X86_64_32S` en `libgenomsa_ref.a`). Flags movidos a scope directorio.
- **Cómo se consiguió slot**: `--gres=gpu:q6000:1 --export=ALL,GENOALIGNER_ARCH=75`
  — las Q6000 (r22rXX) estaban libres; MI210/V100 saturadas. La misma
  receta sirve para A100 (`a100:1`, ARCH=80) y A40/L40 si hace falta.

## Riesgos conocidos

- **`codon_refine_gpu` en device real**: el shim lo ejecuta, pero el path
  de packing puede fallar en device real → el fallback host está
  documentado; el log debe mostrar si ocurrió (el test sigue PASS, la
  salida es idéntica — revisar `align total` lines por si acaso).
- **`hipcc` arch detection en nodo sin GPU visible**: no aplica — los jobs
  piden `--gres` explícito, el arch se detecta en el nodo asignado.
- **cmake fuera de PATH**: `build_rocm.sh`/`build_cuda.sh` ya resuelven la
  ruta `/kuhpc/sw/cmake/3.30.3/...` — heredado.
- **CUDA arch equivocado**: `GENOALIGNER_ARCH` debe coincidir con el GRES
  pedido (70=v100, 75=q6000, 80=a100) — el script valida nvcc pero no el
  par gres/arch; está en el header del sbatch.
