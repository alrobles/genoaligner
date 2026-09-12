# H8-SW1 — SW warp-per-row en GPU: correctitud y determinismo

> **Medido:** 2026-09-12 · jobs **29227061** (fallo de build), **29227073** y
> **29227080** (nodo compartido r06r18n01), **29227111** (diagnóstico de
> contención), **29227117** (nodo exclusivo r07r16n01, MI210)
> **Todos los números están copiados de los logs**, no de memoria.
> Logs: `logs/h8sw1_<job>.out`, `logs/sw_diag_29227111.txt`,
> `logs/smi_sample_29227111.txt`.

---

## 1. Qué se afirma y qué NO

Se afirma, medido en GPU real (MI210, ROCm 6.4.3, warpSize=64):

1. **Correctitud del kernel warp-per-row a `blockDim > 1`.** 210/210 pares
   coinciden con la referencia de matriz completa, en 3 jobs distintos
   (29227073, 29227080, 29227117). Lanzamiento: `grid=53, block=256`
   (4 warps × 64 lanes), smem dinámica 6432 B.
2. **Determinismo del kernel en GPU exclusiva:** 7 lanzamientos idénticos,
   tiempo de device (hipEvent) 2.436–2.442 ms, **dispersión 0.2%**. El
   criterio B6 (< 5%) pasa con dos órdenes de magnitud de margen.

NO se afirma: traceback (no existe), API pública (SW sigue fuera), speedup
sobre la versión de 1 hilo (aún no medido — es el siguiente trabajo), ni
portabilidad NVIDIA (el código usa intrínsecos portables pero no se ha
compilado ni corrido en CUDA).

## 2. Lo que el gate CPU NO pudo ver (y por qué importa correr en GPU)

El primer job (29227061) no compiló: 9 errores, `use of undeclared identifier
'__shfl_up_sync'` y familia. En ROCm ≥ 5.7 los `*_sync` de warp son **opt-in**:
`amd_warp_sync_functions.h` solo los declara si `HIP_ENABLE_WARP_SYNC_BUILTINS`
está definido ANTES de incluir `hip_runtime.h`. El shim los provee siempre, así
que el gate CPU estaba verde sobre un código que hipcc no podía compilar.

Fix: `sw_kernel.hip` define el macro bajo `#if defined(__HIPCC__)` (el macro
predefinido por el driver; `__HIP_PLATFORM_AMD__` no existe todavía a esa
altura — lo definen los propios headers). En el path NVIDIA el header amd_*
no se incluye, así que el macro es inerte; el shim no define `__HIPCC__`.

Lección ya conocida, reconfirmada: el shim valida la álgebra y el flujo de
datos entre lanes; no valida que el código compile contra los headers reales.

## 3. El "fallo" de determinismo era el GPU compartido, no el kernel

Dos corridas en el nodo r06r18n01 mostraron stalls aislados de ~50–400x:

    job 29227073 (reloj wall solamente):
      rep 0: 1000.275 ms   rep 2: 821.102 ms   resto: ~2.45 ms
    job 29227080 (dev = hipEvent, wall = launch->sync):
      rep 4: dev=118.295 ms   resto: ~2.45 ms

El hecho de que el **tiempo de device** también se inflara descarta la espera
en cola como explicación: el kernel fue *preemptado a mitad de ejecución*.

Diagnóstico (job 29227111): se muestreó `rocm-smi --showpids` cada ~200 ms
mientras corría el bench. Todas las muestras muestran lo mismo:

    1034335  agnet-train   GPU 1   4.8 GB
    1034334  agnet-train   GPU 1   4.8 GB
    1038402  sw_gpu        GPU 1   276 MB     <-- nuestro proceso

Slurm asignó `ROCR_VISIBLE_DEVICES=0` a nuestro job, pero ese índice mapea al
mismo GPU físico que usan dos procesos `agnet-train` de otro usuario — el
cluster comparte GPUs por shards (`gres/shard:mi210:300` por nodo). Los stalls
son time-slicing con el co-tenant.

Consecuencia para mediciones pasadas y futuras: **un bench wall-clock en este
cluster mide también al vecino**. El spread de 2.5x que B6 atribuyó al patrón
de barreras del kernel WFA pudo haber estado contaminado por co-tenants — la
interpretación de B6 no se reabre aquí (sus números son los que son), pero el
instrumento correcto para "¿el kernel es determinista?" es el tiempo de device
en GPU exclusiva, y eso es lo que este job mide.

## 4. GPU exclusiva: el número real

Job 29227117, nodo r07r16n01 (estado `idle`, sin otros jobs), misma entrada:

    rep 0: dev=2.439 ms  wall=2.453 ms
    rep 1: dev=2.436 ms  wall=2.450 ms
    rep 2: dev=2.438 ms  wall=2.452 ms
    rep 3: dev=2.440 ms  wall=2.451 ms
    rep 4: dev=2.437 ms  wall=2.448 ms
    rep 5: dev=2.436 ms  wall=2.447 ms
    rep 6: dev=2.442 ms  wall=2.455 ms
    dev spread = 0.2%   ->  PASS (criterio B6: < 5%)

La estructura lo predijo (cero barreras, cada lane solo toca shared propio) y
la medida lo confirma: el schedule del kernel es estable al 0.2%.

## 5. Estado resultante de Fase A

| criterio del plan | estado |
|---|---|
| paridad exacta con `blockDim > 1` | ✅ CPU (110/110 × warpSize 32 y 64) + GPU (210/210 × 3 jobs) |
| el gate falla sin la sincronización | ✅ control negativo: 100/110 fallan con exchange desactivado |
| determinismo, 7 corridas | ✅ dev spread 0.2% en GPU exclusiva |
| speedup con diseño intercalado B6 | ⏳ pendiente — próximo trabajo |

Nota de operación: los jobs de bench que dependan de tiempo deben correr en
nodo sin co-tenants (o medir con hipEvent y reportar ambos relojes, como hace
`test_sw_gpu --bench` ahora).
