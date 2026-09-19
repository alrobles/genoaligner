# Ruta de mejora: backbone filogenético, CASTER y GPU en KUHPC

Fecha de auditoría: 2026-09-18.
Estado: hoja de ruta propuesta; no declara implementadas las fases pendientes.
Alcance: conservar el diagnóstico inicial y contrastar el reporte de avance de CASTER con código, PRs, logs y Slurm. Los estados de jobs son instantáneas de esta fecha, no un monitor en tiempo real. La revisión no modificó código ejecutable, resultados históricos ni jobs; sólo añade documentación.

## 1. Decisión principal

Hay una ruta viable, pero el ejecutor HIP validado todavía no es un CASTER GPU de extremo a extremo. Mantener dos líneas separadas:

1. **Producto científico:** árbol global de referencia con ML y backbone fechado con representantes bien cubiertos, MrBayes/BEAGLE y patches paralelos.
2. **Contribución computacional:** integrar en CASTER el ejecutor HIP residente existente, conservar la semántica estadística, añadir checkpoints y evaluar después concurrencia de guías y multi-GPU.

No seguir optimizando WFA o el NJ del alineador como si eso acelerase automáticamente la inferencia desde una supermatriz ya calculada. No prometer un árbol completo en menos de seis horas a partir de tiempos de microkernels.

Antes de usar RF como gate de equivalencia, corregir el defecto reproducido en la utilidad compartida de RF descrito en la sección 5. Antes de elegir CPU o GPU a partir del benchmark de 3.000 columnas, corregir su interpretación: contiene un solo bloque estadístico.

## 2. Qué significa replicar Upham

Upham et al. (2019) separan tres productos:

- Árbol global RAxML: 4.098 especies, 39.099 posiciones y una selección de nueve particiones. Sirve como scaffold para delimitar grupos y orientar análisis posteriores.
- Backbone fechado: 59 mamíferos más Anolis carolinensis, inferidos con MrBayes y calibraciones fósiles.
- 28 patches: inferencias separadas, reescaladas y ensambladas con el backbone para formar conjuntos de árboles fechados.

Un árbol global de 4.353 terminales no es el backbone reducido y fechado del paper. CASTER estima una topología bajo un objetivo coalescente; no sustituye la datación, las calibraciones ni la distribución posterior de Upham. Tampoco optimiza el mismo objetivo que ML concatenado.

Para la evaluación independiente, Upham permanece **sólo como referencia posterior**, nunca como constraint ni guide. Si se ensaya búsqueda guiada, utilizar árboles inferidos del propio dataset y declarar el cambio de estrategia. Usar el conjunto de nombres compartidos para una comparación no equivale a usar la topología de Upham durante la inferencia.

## 3. Inventario verificado de código, datos y recursos

### Repositorios y despliegue

| Componente | Instantánea revisada | Alcance |
|---|---|---|
| genoaligner público | main, 4b9789ff8c89bfd11995a06f81ef195bdcfb0e05 | WFA/Smith-Waterman pairwise; checkout local limpio |
| genoaligner-devel | main, 7fb5553735761bf427aae969e310f10c41b8922d | MSA y NJ-GPU como guía |
| genoaligner-devel | devin/codon-msa, 08ccc367cd7b7d650af931b8d9fe159d6742bdcb | Rama activa de pipelines y CASTER; PRs 7, 8 y 9 integrados aquí, no en main |
| alrobles/ASTER | caster-v1.25-base, 1624214eca09a57cb1155e84493da988a373eac1 | Prototipo y ejecutor residente; no confundir con master |
| ASTER CPU de producción | db2b3e95da5bb0318b933afe1a144eb943ef7cbf, tag v1.25 | Driver caster-site, versión impresa v1.25.2.6 |

La conexión Hermes a KUHPC funcionó como a474r867. Rutas relevantes:

- `/beegfs/a474r867/genoaligner/repo`: f31b5dfa12dc167513671546af1f52d1ce4a3e5d; tres commits detrás de la rama activa revisada y con archivos no versionados. No actualizar a ciegas un checkout usado por jobs.
- `/beegfs/a474r867/genoaligner/caster-benchmark-resume`: checkout limpio en 08ccc367cd7b7d650af931b8d9fe159d6742bdcb.
- `/beegfs/a474r867/phylogenyAI`: matrices, resultados y herramientas CASTER CPU.
- `/kuhpc/work/kbs/a474r867/Github/ASTER`: checkout del acelerador en 022c142706b1088746eb9a00918e06af6b582c32, con el mismo contenido que la rama integrada revisada. Este código está en NFS, no en BeeGFS.
- `/beegfs/a474r867/ASTER/images`: existen caster-rocm.sif y caster-cuda.sif.

### Matrices disponibles

Dimensiones contrastadas con las tablas de cobertura, las particiones y el primer registro FASTA; esto no sustituye una auditoría exhaustiva del alineamiento:

| Directorio en phylogenyAI/data | Filas declaradas | Columnas | Particiones por locus |
|---|---:|---:|---:|
| supermatrix | 4.353 | 76.475 | 31 |
| supermatrix_genomsa | 4.353 | 280.189 | 31 |
| supermatrix_genomsa_lf | 4.353 | 240.125 | 31 |
| supermatrix_codon | 4.353 | 247.185 | 31 |

No identificar automáticamente las 4.353 filas con especies válidas. En `supermatrix_genomsa_lf/taxon_coverage.tsv`:

- Mediana de genes_present: 3; 1.340 filas declaran un solo gen.
- 76 filas declaran al menos 25 genes.
- 68 etiquetas tienen formato de accession; también aparece Hamster_mRNA.

Estos conteos se refieren a presencia declarada de genes, no al porcentaje de bases observadas. Reconciliar nombres y copias antes de descartar datos o elegir representantes. Seleccionar representantes por cobertura **y diversidad de clados**, no sólo por un ranking de cobertura.

Hay una discrepancia de procedencia: `results/supermatrix_tree_genomsa/backbone.log` registra 96.718 columnas, pero la matriz actualmente en la ruta que ese log declara tiene 280.189. Aclarar qué versión produjo cada árbol/checkpoint antes de interpretar comparaciones. El nombre del archivo no identifica su contenido.

### GPU y límites

Conteos de GPU configuradas en nodos únicos de sixhour, contrastados con knowledgebase; no son GPU libres ni capacidad reservable simultáneamente:

| Tipo | Cantidad |
|---|---:|
| MI210 | 81 |
| V100 | 36 |
| Q6000 | 29 |
| A100 | 20 |
| PRO6000 | 5 |
| A40 | 4 |
| L40 | 4 |
| Q8000 | 2 |

- sixhour: límite efectivo 06:00:00. Las MI210 están distribuidas en 27 nodos de tres tarjetas.
- kbs: cuatro nodos CPU, sin GPU declaradas; máximo configurado 60 días. Los CASTER completos pidieron siete días.
- No asumir que todas las A100 son de 80 GB: la validación del acelerador registró una A100-PCIE-40GB.
- Los fallos históricos con nueve o más ranks RCCL pertenecen al stack concreto documentado en knowledgebase, no son un límite universal del hardware o de MPI.
- BeeGFS permite compartir artefactos entre jobs; no asumir por ello una política de backup o retención que esta auditoría no verificó.

### Resultados existentes

En el corte inicial de la auditoría:

- CASTER mini: 13/15 árboles completos; faltaban rep1/base y rep1/macse. Los completos registraban 19–94 s, mediana 37 s.
- CASTER completo: dos tareas de 29620811 en ejecución durante más de 24 h y una pendiente; aún sin árboles finales publicados.
- IQ-TREE lf: cuatro cadenas con checkpoint, contador de enlace 10; no confundir los treefiles intermedios con resultados finales y soporte convergido.
- FastTree: tres árboles completos con TreeCompleted; tiempos registrados 40.303,06 s (genomsa), 34.324,36 s (lf) y 33.205,83 s (codon). Son guías propias disponibles, no una demostración de ejecución menor a seis horas.

No se recalcularon los RF históricos ni se promovieron sus valores a evidencia definitiva; revisar su validez tras corregir la utilidad RF y fijar la procedencia de las matrices.

## 4. Dictamen del reporte de avance CASTER

### Infraestructura reproducible: avance real, con límites de despliegue

Los PRs 7–9 de genoaligner-devel implementan builds portable/strict, arrays, selección de celdas faltantes, exclusión por flock, metadata ante TERM/INT/HUP, reportes afterany y comparación sobre terminales compartidos.

Esto no equivale todavía a procedencia completa de cada ejecución:

- La metadata activa de `cpu-scale-3000/t1` incluye hash del binario, pero deja vacíos perfil, commit ASTER, compilador y flags.
- El runner registra tamaño de entrada, no un hash de la matriz.
- La detección de resultado reutilizable comprueba archivo no vacío y presencia de `;`; no valida manifiesto, identidad de entrada o conjunto completo de terminales.
- Los traps no garantizan publicar metadata bajo SIGKILL o fallo del nodo. El benchmark anterior terminó por timeout y conservó status=running.

Son pendientes de endurecimiento, no motivos para descartar la infraestructura entregada.

### GPU heterogénea: confirmado para replay, no para árbol completo

- Jobs 29645269 y 29645270: validation=passed en MI210 y A100-PCIE-40GB, con el fixture de 101 terminales, tres particiones y 702 sitios retenidos.
- Job 29645003: replay MI210 sobre los 4.353 terminales, 29 particiones y 31.933 sitios retenidos; asignación de dispositivo 140.920.249 bytes.
- La evidencia A100 revisada no demuestra todavía el replay del dataset completo de 4.353 terminales en esa tarjeta.
- La fase 1 usa entradas del parser de producción, pero su cinta de movimientos es generada para validación, no una traza completa de decisiones de búsqueda.
- El ejecutor HIP no decide todavía la topología de producción. Cuadriparticiones, soporte integrado, recuperación y multi-GPU siguen pendientes.
- El comparador executeProduction del replay recorre las partes CPU secuencialmente: su tiempo no es un benchmark contra una búsqueda CPU optimizada de cuatro hilos.

### CPU: 3,04x y 82,6 % son cifras rastreables, no generalizables

El job 29632877 midió una repetición por configuración sobre el mini de 147 terminales y 47.745 columnas, en r05r10n03:

| Hilos | Tiempo registrado |
|---|---:|
| 1 | 296,508 s |
| 2 | 190,435 s |
| 4 | 97,577 s |
| 8 | 98,587 s |
| 16 | 99,759 s |
| 32 | 103,649 s |

El cociente 1/4 hilos es aproximadamente 3,04x. Cuatro hilos es el mejor punto observado en esa medición, no una elección validada para 4.353 terminales. No hay repeticiones suficientes para estimar dispersión.

El 82,6 % se identifica en `gprof_nopie/gprof_report.txt`: 66,6 % inclusivo de updateCnt y 16,0 % de scoreCnt. No volver a sumar sus callees: se duplicaría tiempo atribuido.

El perfil corresponde al job 29632917, con 50 terminales, un hilo y opciones `-r 1 -s 0 -u 0`; compilación `-O2 -pg -no-pie -fno-inline -pthread`. No es el binario portable de producción `-O3 -ffast-math`. Además, n=50 no entra en la ruta two-step activada desde 100 terminales, y se excluyeron subsampling y soporte. El intento perf del job 29632873 registró PERF_UNAVAILABLE.

Conclusión: el perfil orienta hacia updates/scoring, pero el 82,6 % no es una fracción medida del walltime completo ni permite proyectar un speedup global mediante Amdahl sin otra medición.

### 796 guías y cola residual: confirmados con precisión de alcance

En `twoStepWorkflow`, el tamaño se calcula como:

```text
n = floor(sqrt(N) * log2(N) / 4)
N = 4353 -> n = 199
4 rondas iniciales -> 4 * 199 = 796 guías
```

Se contaron 796 registros Guide Tree antes de Subsample Process en el log del job 29907445: cuatro comienzos y cuatro finales de grupos de 199. Son guías sobre subconjuntos de 199 terminales, **no 796 búsquedas completas de 4.353 terminales**.

Las colas Remaining de las cuatro rondas iniciales fueron 1.296, 1.122, 1.098 y 1.516; la primera observada en subsampling fue 1.339. Son observaciones de esta entrada/semilla, no constantes del algoritmo.

La propuesta `guías -> ramas -> sitios` es plausible, pero su exactitud aún no está demostrada:

- Tripartition comparte TripartitionInitializer; Gene contiene contadores mutables y caches, y una copia superficial conserva el shared_ptr de los contadores.
- Se comparten ThreadPool y estado aleatorio. No basta con añadir un bucle paralelo.
- Cada ronda exterior incorpora triparticiones al DAG. No tratar las 796 guías como una única bolsa de tareas independiente de las barreras actuales.
- Los grupos nodeBranch reutilizan nodos/contadores y ensamblan pNodes y abnormalOrder. Una fusión por orden de finalización puede cambiar la cola residual y la trayectoria posterior.

### Benchmark de 3.000 columnas: útil para control, inválido para elegir hilos de producción

El log activo registra `#Species: 4353` y `#Base: 1471314`: 1.471.314 / 4.353 = **338 sitios retenidos por el filtro de CASTER**. No confundir este término con todos los sitios informativos según parsimonia.

Se completaron las cuatro fases iniciales de guías y comenzó subsampling. En la segunda revisión no había Final Tree ni tiempo final, RF o memoria final disponibles. No se reejecutó el benchmark para obtener estas observaciones.

El parser usa `--chunk 10000` y el reparto CPU opera sobre ventanas estadísticas completas:

| Columnas de entrada | Ventanas creadas |
|---|---:|
| 3.000 | 1 |
| 30.000 | 3 |
| 47.745 | 5 |
| 280.189 | 29 |

Con una ventana, sólo un trabajador recibe el trabajo de sitios, incluso solicitando 32 hilos. Completar la tabla 1–32 no resolverá ese defecto de diseño. La muestra sí puede servir como prueba del control de búsqueda con muchos terminales, pero no representa la granularidad ni la carga de sitios de producción.

No reducir chunk para aparentar mejor escalado: modifica frecuencias y unidades del bootstrap. Separar ventanas estadísticas de tiles computacionales.

La cautela sobre comparar 30.000 y 3.000 columnas en nodos distintos es correcta. Los jobs de respaldo 29907471–29907476 esperan afterany:29907445 y apuntan a las mismas celdas; son submissions redundantes de recuperación, no evidencia de búsquedas duplicadas simultáneas. No se canceló ni añadió ninguna tarea durante esta revisión.

## 5. Bloqueo prioritario: RF depende indebidamente del enraizamiento

Reproducción en memoria usando la utilidad desplegada, sin escribir fixtures ni cambiar resultados:

```text
A = (D,(A,(B,C)));
B = (A,(D,(B,C)));
C = ((A,D),(B,C));

RF(A,B): observado 1, RF_norm 0,333333; esperado 0
RF(A,C): observado 1, RF_norm 0,333333; esperado 0
RF(B,C): observado 0; esperado 0
```

Los tres representan el mismo árbol no enraizado: su única bipartición no trivial es `{B,C}|{A,D}`. La utilidad añade a A la partición trivial `{A,B,C}|{D}`.

Causa en [rf_distance.py](../scripts/rf_distance.py), tanto en splits como en project: después de canonicalizar con `min(mask, complement)`, se comprueba que el lado seleccionado no tenga un solo bit. El mínimo numérico no garantiza seleccionar el lado de menor cardinalidad; puede conservar n-1 terminales frente al singleton del bit más alto.

Este defecto ya estaba en el commit 4d7a0fb8cb687e65f013940437b4679404117749, anterior a los PRs CASTER 7–9; no se atribuye su introducción a esos PRs.

Acción pendiente:

1. Excluir biparticiones si cualquiera de sus lados tiene menos de dos terminales, en ambas rutas.
2. Añadir regresiones de reenraizamiento, poda a terminales compartidos, permutación de etiquetas y politomías, conservando el control de cuartetos diferentes RF=2.
3. Contrastar con una implementación independiente y validar nombres duplicados o entradas malformadas.
4. Sólo entonces regenerar las comparaciones afectadas sobre snapshots congelados y versionar sus nuevos reportes.

No se modificó el algoritmo de RF en esta tarea. No asumir que todos los RF históricos son incorrectos ni que los efectos tienen la misma magnitud; su alcance debe medirse después de la corrección. Mientras tanto, no usar la utilidad como gate general de equivalencia no enraizada.

## 6. Ruta de implementación priorizada

### P0. Métricas, datos y procedencia

Entregables:

- Corrección y regresiones de RF de la sección 5.
- Manifiesto inmutable de matriz, particiones, terminales/mapeo, semilla, parámetros, commit, compilador, flags y binario; hashes de contenidos, no sólo tamaños.
- Reconciliación taxonómica y aclaración del log de 96.718 columnas frente al archivo actual de 280.189.
- Validación de un resultado reutilizable contra su manifiesto y terminales, no sólo un punto y coma.

Criterio de salida: RF invariante al reenraizamiento sobre el mismo árbol no enraizado; cada árbol y checkpoint se puede vincular a una entrada identificada. No alterar automáticamente los datos o jobs ya existentes.

### P1. Baseline CPU y trazas representativas

Conservar la muestra de 3.000 columnas como prueba de control, no como selector de CPU. Capturar trazas reales de inserción, NNI y soporte con las ventanas de producción intactas, y congelarlas antes de repetir mediciones.

Medir por separado parsing, selección de guías, clasificación/colocación por ramas, cola residual, scoring, soporte, transferencias y checkpoint. Registrar ventanas/tiles no vacíos por trabajador, hardware, afinidad y flags. Comparar configuraciones intercaladas dentro de la misma asignación y repetir; no inferir speedup formal desde distintos nodos.

Criterio de salida: perfil de la ruta relevante de producción y separación explícita entre aceleración de kernel, replay y búsqueda completa. No extrapolar el perfil instrumentado de 50 terminales a la búsqueda de 4.353.

### P2. Scheduler determinista de guías y ramas

Primera unidad candidata: las 199 guías de una fase inicial, sobre un snapshot común. Precalcular los órdenes aleatorios respetando el consumo serial original, aislar estado mutable y fusionar las triparticiones por el orden lógico original. Mantener las barreras entre fases/rondas salvo prueba de independencia.

Cada tarea necesita contadores, colores, caches, cola de operaciones y nodos privados; sólo comparte datos inmutables. Un shared_ptr copiado no aísla el estado. Evitar sobreasignación multiplicando indiscriminadamente workers exteriores e hilos interiores.

Para nodeBranch, producir resultados con índices locales y fusionarlos de forma determinista, preservando también abnormalOrder. El futuro kernel `rama x sitio` debe evaluar el mismo conjunto de candidatos sobre el mismo estado que la referencia. Evaluar más ramas o aplicar varios NNI simultáneamente puede cambiar la heurística y debe ser un experimento separado.

Criterio de salida: equivalencia contra la referencia serial con una y varias tareas, independientemente del orden de finalización, incluida la cola residual y los empates. La ausencia de data races no basta para probar equivalencia científica.

### P3. Integración HIP en la búsqueda real

Reutilizar ResidentDataset, TapeBatch y PortableHipExecutor del fork ASTER. La CPU conserva topología, candidatos, decisiones y criterios de parada; la GPU mantiene observaciones y contadores residentes y devuelve scores para las decisiones reales.

Conservar frecuencias, pesos, unidades de bootstrap y orden de operaciones. Dividir ejecución en tiles sin reconstruir ventanas estadísticas. Incluir la fase de cuadriparticiones/soporte, o medir explícitamente su ejecución CPU; no declarar acelerado todo CASTER por acelerar sólo triparticiones.

Criterio de salida: contadores exactos, scores y decisiones validados, árboles completos equivalentes en los casos de prueba y backend real registrado. Extender la validación del dataset completo a NVIDIA antes de afirmar cobertura completa en ambos proveedores. Tolerancia de score por sí sola no demuestra RF cero.

### P4. Continuación real entre ventanas de seis horas

Implementar checkpoints del estado de búsqueda, no sólo del mejor Newick: fase/progreso, candidatos/DAG, órdenes, generadores aleatorios, parámetros y vínculo al manifiesto. Los contadores GPU pueden reconstruirse desde un estado CPU suficiente y validado.

Crear puntos seguros dentro de inserciones y NNI, no únicamente al terminar una ronda potencialmente larga. Vaciar la cola pendiente antes de publicar el checkpoint, usar publicación atómica y conservar el último estado válido.

El launcher debe guardar antes del límite con margen, excluir escritores concurrentes, verificar avance/checkpoint antes de reenviar y limitar reintentos. Un timeout, un error persistente y una finalización correcta requieren decisiones diferentes.

Criterio de salida: interrupción y reanudación equivalentes a ejecución continua, incluidas recuperación desde escritura incompleta y negativa a reanudar con otra matriz o parámetros incompatibles. Encadenar el CASTER actual sin este estado sólo repite trabajo.

### P5. Multi-GPU después de validar una GPU

Comparar una MI210, dos/tres MI210 del mismo nodo y búsquedas independientes por tarjeta. La memoria observada en el replay no obliga a distribuir datos; el potencial beneficio está en concurrencia y utilización, no en una necesidad demostrada de VRAM.

Opciones distintas:

- Guías o búsquedas independientes por GPU: alto paralelismo con poca comunicación; mantener el orden de fusión si se pretende reproducir la búsqueda serial.
- Tiles de sitios para evaluar el mismo candidato: sumas parciales con frecuencias/pesos originales y reducción reproducible.
- Subárboles de grupos de taxones: un pipeline inferencial diferente, no una reducción exacta de scores.

No usar BeeGFS como canal por cada score. Medir antes de pasar a varios nodos o mezclar proveedores en una ejecución acoplada.

Criterio de salida: mejora de tiempo total o throughput bajo un presupuesto comparable, incluyendo transferencias, sincronización, relectura y checkpoints. No confundir varios árboles producidos en paralelo con aceleración de una búsqueda individual.

### P6. Validación científica y producto

Separar equivalencia numérica CPU/GPU de comparación entre estimadores. Comparar RF corregido sobre los mismos terminales, concordancia, soporte, estabilidad entre réplicas y sensibilidad a cobertura. Contrastar nuclear frente a mitocondrial sin tratar automáticamente fragmentos ligados como loci independientes.

Comparar log-likelihood sólo dentro de la misma matriz y modelo, optimizando los parámetros que correspondan. No considerar cercanía a Upham como verdad de referencia ni usar su topología para ajustar el método que luego se compara con él.

Criterio de salida: árbol global o backbone fechado identificado explícitamente, cobertura taxonómica documentada, soporte/convergencia adecuados al método, entradas trazables y tiempo de extremo a extremo medido.

## 7. Alternativas que mantienen viable el producto científico

| Ruta | Papel | Limitación principal |
|---|---|---|
| MrBayes + BEAGLE | Backbone fechado de representantes y patches; GPU para likelihood, réplicas independientes y checkpoints | Validar BEAGLE/CUDA; AMD/OpenCL requiere comprobar el entorno; no asumir que HIP sirve directamente |
| IQ-TREE / RAxML-NG | Referencia ML del árbol global | CPU y checkpoints; RAxML-NG admite MPI; no presuponer GPU de producción |
| CASTER CPU guiado por árboles propios | Reutilizar FastTree para explorar localmente | Heurística guiada declarada; un guide no restaura el estado de búsqueda |
| CASTER HIP integrado | Línea de desarrollo principal | Aún faltan búsqueda completa, recuperación y validación final |
| Árboles por locus + ASTRAL/wASTRAL | Contraste coalescente con loci paralelos | Sólo 31 loci y cobertura desigual; error de árboles génicos y ligamiento |
| VeryFastTree | Baseline rápido/guías | CPU paralela; CUDA experimental, sin garantía medida para este dataset |
| TreeMerge/GTM | Divide-and-conquer por taxones | Necesita información global para unir subárboles; puede fijar errores locales |

El NJ-GPU existente es útil como guía, no reemplaza ML, inferencia coalescente o datación. No priorizar una nueva vía variacional/hiperbólica antes de cerrar los bloqueos medidos del pipeline existente.

## 8. Fuentes y artefactos de la auditoría

Código revisado:

- [Experimento CASTER](CASTER_BACKBONE.md), [runner](../scripts/caster_run.sh), [benchmark CPU](../scripts/caster_cpu_scale.sbatch), [reporte](../scripts/caster_report.py), [RF](../scripts/rf_distance.py) y [tests](../tests/test_caster_report.py).
- [genoaligner-devel PR 7](https://github.com/alrobles/genoaligner-devel/pull/7), [PR 8](https://github.com/alrobles/genoaligner-devel/pull/8), [PR 9](https://github.com/alrobles/genoaligner-devel/pull/9).
- [ASTER PR 1](https://github.com/alrobles/ASTER/pull/1), [PR 2](https://github.com/alrobles/ASTER/pull/2), [PR 3](https://github.com/alrobles/ASTER/pull/3), [PR 4](https://github.com/alrobles/ASTER/pull/4).
- ASTER v1.25: src/caster-site.cpp (ventanas/frecuencias), src/sequence.hpp (estado/reparto), src/algorithms.hpp (twoStepWorkflow, twoStepRun y decisiones) y src/threadpool.hpp.
- Fork ASTER: src/caster-site-accelerator-bench.cpp, src/caster-site-accelerator-data.hpp, src/caster-site-accelerator-hip.hpp y misc/caster-accelerator-phase1.md.

Artefactos HPC, leídos sin volver a ejecutar las inferencias:

- `/beegfs/a474r867/phylogenyAI/results/caster_gpu_assessment/cpu_scaling_v2/timings.tsv`.
- `/beegfs/a474r867/phylogenyAI/results/caster_gpu_assessment/gprof_nopie/gprof_report.txt` y gprof.log; receta del job 29632917 obtenida de sacct.
- `/beegfs/a474r867/phylogenyAI/results/caster_backbone/benchmark/cpu-scale-3000/t1/caster.log.tmp` y run.meta.tsv, job 29907445. El log estaba activo: los conteos citados son una captura, no un resultado final.
- `/beegfs/a474r867/phylogenyAI/results/caster_backbone/report/caster_runs.tsv` y las celdas mini_v2.
- `/home/a474r867/scratch/caster-p1-real-29645003.out`, caster-ctr-amd-29645269.out y caster-ctr-nv-29645270.out.
- squeue/sacct/scontrol/sinfo para estados, recetas, límites e inventario; knowledgebase/hpc-scripts/ollama/INFERENCE_DEPLOYMENT.md y hpc-scripts/multi-node-gpu-training.md para antecedentes.

Referencias metodológicas:

- [Upham et al. 2019](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000494).
- [CASTER: Direct species tree inference from whole-genome alignments](https://pmc.ncbi.nlm.nih.gov/articles/PMC12038793/).
- [IQ-TREE: checkpoints](https://iqtree.github.io/doc/Command-Reference).
- [RAxML-NG: checkpoints](https://codeberg.org/amkozlov/raxml-ng/wiki/Advanced-Tutorial) y [paralelización](https://codeberg.org/amkozlov/raxml-ng/wiki/Parallelization).
- [MrBayes/BEAGLE y reanudación](https://hpc.nih.gov/apps/mrbayes.html).
- [VeryFastTree](https://github.com/citiususc/veryfasttree/).
- [TreeMerge](https://pmc.ncbi.nlm.nih.gov/articles/PMC6612878/).

## 9. Próxima acción recomendada

Corregir primero RF y la procedencia de las entradas; separar el benchmark de control del benchmark de sitios. Después cerrar la especificación y el baseline serial del scheduler determinista, integrar HIP con una GPU y checkpoints, y sólo entonces evaluar multi-GPU y el backbone completo optimizado.

La documentación de esta ruta no autoriza cancelar jobs, sobrescribir matrices/checkpoints, publicar resultados como finales ni lanzar nuevas inferencias. Esas acciones requieren su propio paso de ejecución y revisión.

## 10. Plan experimental progresivo y modificaciones

Estado: protocolo propuesto el 2026-09-18; documentado, todavía no implementado ni ejecutado. Las cantidades de réplicas, semillas, presupuestos y umbrales de esta sección son decisiones de diseño para la siguiente campaña, no resultados medidos. No reemplazan ni reinterpretan retrospectivamente los números de la auditoría.

### 10.1 Principio de trabajo

La secuencia será **detectar -> simular de forma controlada -> medir -> proponer una modificación -> contrastar -> confirmar con datos nuevos -> aumentar escala**. Un patrón ausente o inconcluso es un resultado válido; no aumentar réplicas o cambiar casos hasta obtener un resultado favorable.

El principio técnico ya identificado es que **ventana estadística no equivale a unidad de ejecución**. Preservar las ventanas de CASTER y sus frecuencias/bootstraps, y modificar únicamente la distribución del trabajo cuando se pretenda una aceleración equivalente.

Separar dos preguntas:

1. **Ingeniería:** ¿la versión optimizada resuelve la misma tarea, con el mismo resultado de referencia, en menos tiempo o con mayor throughput?
2. **Inferencia:** ¿qué relación tienen cantidad de señal, cobertura y tamaño del problema con el error frente a un árbol verdadero simulado?

RF CPU/GPU cero no demuestra que el árbol sea biológicamente correcto. Menor RF frente a la verdad no demuestra que dos implementaciones hagan el mismo trabajo. Mantener ambas evaluaciones separadas.

### 10.2 Infraestructura que se reutiliza y límites comprobados

- `scripts/codon_sim_v2.py` emite alineamiento verdadero, árbol verdadero, eventos y manifiesto; `scripts/msa_sim_truth.py` también produce verdad de homología. Ambos usan árboles balanceados y no implementan un modelo multiespecífico coalescente con ILS.
- `scripts/codon_bench_grid.py` y `scripts/sim_precision.sbatch` son infraestructura de evaluación de alineadores con FastTree, no un benchmark directo de inferencia CASTER. Reutilizar su organización por celda/réplica, no confundir SPS/TC del alineamiento con exactitud del backbone.
- Para esta campaña, generar una vez cada entrada y conservar `sim_true.fasta`, árbol y manifiesto por hash. El análisis posterior debe leer esos artefactos; no regenerar la verdad cada vez que se puntúa otro backend.
- Se comprobó que `/beegfs/a474r867/miniconda3/envs/genoml-phylo/bin/iqtree2`, versión 2.4.0, anuncia AliSim y sus opciones de simulación. AliSim será el generador independiente principal del panel inicial, sin volver a alinear sus secuencias.
- En ese entorno se localizaron NumPy y SciPy; no se localizaron DendroPy ni msprime. No asumirlos instalados. Preparar dependencias adicionales en un entorno aislado, con versiones fijadas y publicadas al menos siete días antes, únicamente cuando se apruebe esa fase.
- La workflow CPU actual ejecuta `scripts/check_kernel_cpu.sh`, pero no la suite Python `tests/test_caster_report.py`. Incorporar esa suite al gate de CASTER para que las regresiones RF no dependan de ejecución manual.

### 10.3 Variables e hipótesis contrastables

Registrar en cada celda:

| Símbolo | Significado |
|---|---|
| N | Terminales esperados y terminales realmente presentes |
| L | Columnas originales del alineamiento |
| S | Sitios retenidos por el parser de CASTER |
| K | Ventanas estadísticas totales y ventanas con sitios retenidos |
| P | Hilos CPU asignados y trabajadores que reciben trabajo efectivo |
| G | Guías ejecutadas, tamaño de cada guía y fase/ronda |
| U, Q | Actualizaciones lógicas y evaluaciones de score; registrar también las evitadas por cache |
| B | Candidatos u operaciones realmente independientes agrupados por batch |
| A | Longitud de abnormalOrder/Remaining y tiempo de su procesamiento |
| M | Ausencia de datos observada, por sitios y por taxón/locus |

Hipótesis de trabajo:

- **H1, granularidad:** la meseta CPU depende de K y del trabajo efectivo por ventana; aumentar P por encima de las unidades disponibles no crea paralelismo de sitios.
- **H2, control serial:** guías, clasificación por ramas y cola residual explican una parte del walltime no capturada por un microbenchmark de score. Su fracción temporal debe medirse, no inferirse de cuántos terminales quedan.
- **H3, amortización GPU:** existe un régimen de S y B en que el trabajo residente amortiza lanzamientos, transferencias y reducción; por debajo puede ser mejor CPU. No asumir que ese cruce depende sólo de N o de L.
- **H4, información:** el error frente al árbol verdadero depende de señal y del patrón de datos faltantes; añadir columnas no necesariamente recupera relaciones que carecen de solapamiento informativo.

CASTER cambia a la ruta two-step desde N=100. Incluir expresamente N=99, 100 y 101 y tratar ese cambio como un régimen distinto, no ajustar ciegamente una única ley de potencia a todos los tamaños.

### 10.4 Escalera de simulaciones

No ejecutar el producto cartesiano de todos los factores. Cada escalón tiene una pregunta, un presupuesto y una condición de avance.

| Escalón | Casos propuestos | Objetivo y salida |
|---|---|---|
| E0: instrumentos | Árboles manuales de 4–16 terminales; fixtures pequeños de contadores/scores | Oráculos capaces de detectar errores antes de medir rendimiento |
| E1: granularidad | Panel de 11 celdas descrito abajo, inicialmente balanceadas, JC, sin gaps ni indels | Separar N, L, K y P; medir cambios de ruta y captura de trazas reales |
| E2: dificultad | N=128, L=40.000; tres formas de árbol y tres mecanismos de ausencia | Distinguir rendimiento, error del estimador y sesgo del simulador |
| E3: una modificación | Misma entrada y trazas congeladas; cambiar una sola característica por comparación | Aislar el efecto de tiles, scheduler o integración HIP |
| E4: escala | N=512 -> 1.024 -> 2.048 -> 4.353; una matriz real versionada al final | Pasar de replay a búsquedas completas reanudables sin saltar directamente al caso mayor |
| E5: confirmación | Variante, CPU de comparación y casos primarios congelados; semillas no usadas en exploración | Confirmar o rechazar el patrón sin reutilizar los datos que motivaron la modificación |

**E0, exactitud primero.** Añadir las invariancias RF de la sección 5, incluido el caso que hoy falla, la poda a terminales compartidos, permutaciones de etiquetas y estrella frente a cuarteto resuelto. Conservar los casos negativos y los denominadores; un archivo vacío o una comparación sin terminales suficientes no constituye éxito.

Para scores, mantener inicialmente la tolerancia existente del ejecutor:

```text
abs(cpu - candidate) <= 1e-6 + 1e-12 * max(abs(cpu), abs(candidate))
```

Antes de aplicar esa fórmula, exigir operandos finitos. El validateScores revisado no contiene esa comprobación explícita: comparaciones con NaN/Inf pueden no activar su condición de fallo. Añadir pruebas negativas para NaN e infinitos y confirmar que hay scores realmente comparados en el experimento completo. Un batch legítimo de sólo updates puede no emitir scores; no confundirlo con un experimento vacío.

Mantener contadores idénticos y verificar decisiones/topologías, además de la tolerancia escalar. No aflojar la tolerancia para hacer pasar una GPU. Ante diferencias en decisiones cercanas a un empate, investigar reducción, aritmética y desempate antes de promover el backend.

**E1, panel inicial de 11 celdas.** Siempre un solo FASTA y `--chunk 10000`:

- N=128; L en {3.000, 10.000, 40.000, 160.000, 320.000}: cinco celdas.
- L=40.000; N en {32, 60, 99, 100, 101, 256}: seis celdas adicionales.
- Las longitudes generan K={1, 1, 4, 16, 32} en el primer barrido. Registrar por separado cuántas ventanas retienen sitios; no confundir K con los 31 loci biológicos.
- Árboles balanceados, escala máxima raíz-punta de 0,2 sustituciones/sitio, JC, sin indels, una secuencia por terminal. Guardar el Newick con longitudes usado por AliSim.
- La verdad se usa sólo para evaluar; no pasar ese árbol a CASTER mediante guide o constraint.
- Una referencia serial completa por dataset cuando quepa en el presupuesto permite capturar operaciones y fases. Las trazas se registran en la frontera lógica, no una vez por cada trabajador, para no multiplicar artificialmente U y Q por P.
- En el replay, probar P={1,2,4,8,16,32}. No multiplicar automáticamente todas las búsquedas completas por todos los P: reservarlas para la referencia y las configuraciones seleccionadas antes de confirmación.
- La celda de 3.000 columnas conserva su papel de control negativo de granularidad. No descartarla porque no escale y no usarla para elegir la CPU de producción.

**E2, panel de dificultad separado.** Mantener N=128 y L=40.000, usando árboles balanceados, pectinados y aleatorios Yule (nacimiento puro, condicionado a N terminales). Fijar inicialmente longitudes unitarias en las formas deterministas y conservar las longitudes del generador Yule; después normalizar cada árbol a máxima distancia raíz-punta de 0,2. Guardar el árbol resultante y los parámetros del generador. Registrar también la distribución de longitudes internas: igual altura no implica igual dificultad topológica.

Usar tres condiciones de ausencia: ninguna; enmascaramiento independiente taxón/sitio con probabilidad 0,8; y enmascaramiento taxón/locus con probabilidad 0,8 sobre 31 bloques de locus. Esas probabilidades son nominales; registrar la ausencia efectiva y su conectividad. Mantener las máscaras congeladas entre backends. No eliminar ni regenerar silenciosamente terminales sin observaciones; registrar entradas no identificables o inválidas como tales, con su denominador.

Usar JC en este primer panel para controlar el modelo. Extensiones a heterogeneidad de tasas, composiciones o ILS tendrán un manifiesto y parámetros propios antes de ejecutarse. AliSim sobre un único árbol no simula ILS: una fase MSC requiere un generador validado de genealogías, unidades demográficas explícitas y su correspondiente árbol de especies. La ausencia actual de msprime/DendroPy debe resolverse antes de presentar esa fase como disponible.

**E3, modificaciones aisladas.** Comparar separadamente:

1. Subdivisión computacional de ventanas, sin alterar frecuencias ni bootstrap.
2. Guías concurrentes con estado privado y fusión determinista por grupo/fase.
3. Integración del ejecutor HIP residente en la búsqueda real.
4. Evaluación rama x sitio conservando candidatos y decisiones de la referencia.

No activar simultáneamente las cuatro y atribuir el resultado a una de ellas. Aplicar varios NNI simultáneamente queda fuera de la equivalencia estricta; si se estudia, será una variante heurística y un experimento independiente.

**E4, escala y continuidad.** Para los tamaños mayores empezar con trazas acotadas, identificando fase y posición muestreada. No presentar una traza parcial como búsqueda completa. Las búsquedas completas que no quepan requieren primero los checkpoints de P4; no encadenar reinicios desde cero. La matriz real y los subconjuntos se seleccionan y congelan una vez después de resolver P0, sin reextraerlos en cada reporte.

### 10.5 Réplicas, presupuesto y separación exploración/confirmación

Configuración propuesta para la primera campaña:

- Exploración: semillas de datos 11001, 11002 y 11003; tres repeticiones técnicas pareadas por comparación y dataset en el replay.
- Confirmación: semillas de datos 21001–21010, reservadas; cinco repeticiones técnicas pareadas por comparación y dataset cuando el caso completo quepa en el presupuesto.
- Semilla de búsqueda CASTER: 233, igual entre implementaciones sobre el mismo dataset. La semilla de datos y la semilla de búsqueda son campos distintos. Registrar además subsemillas explícitas para topología, evolución y máscara, sin depender del hash aleatorio de procesos Python. Un estudio de robustez frente a semillas de búsqueda será otra dimensión, no ruido técnico.
- Orden A/B o B/A prefijado en el manifiesto con semilla de orden 31001, balanceado entre bloques con diferencia de conteos como máximo uno. Emparejar dentro de la misma asignación/nodo; no construir speedups a partir de jobs de hardware distinto.
- Una repetición de calentamiento no puntuada para los microbenchmarks residentes; registrar aparte ejecución fría de la aplicación, carga estática y tiempo total. No presentar tiempo residente como tiempo total del árbol.
- Antes de cada repetición restaurar exactamente el snapshot inicial de contadores, colores, caches y estado aleatorio de ambos comparadores. No reutilizar por accidente el estado final de la repetición anterior. Registrar el coste de restauración por separado y declarar si pertenece al camino de producción medido; no cargarlo sólo a uno de los backends.
- Límite exploratorio por búsqueda completa: 1.200 segundos. Es un presupuesto para detectar fronteras de viabilidad, no una predicción de duración. No hacer reintentos automáticos de una celda que agota su presupuesto sin cambiar de fase/protocolo.
- Jobs experimentales en sixhour: solicitar como máximo 05:50:00; dejar margen operativo. Para ejecuciones reanudables, objetivo de guardado/cierre a 05:30:00. No iniciar otro bloque si no cabe en el presupuesto restante.
- No lanzar el panel entero automáticamente: ejecutar primero E0 y después una celda de E1, verificar artefactos, y ampliar sólo tras revisar el lote anterior.

Antes de abrir las semillas de confirmación, congelar la variante candidata, su política de fallback, los casos primarios, el presupuesto y la mejor configuración CPU válida observada en exploración. La selección de esa CPU debe registrarse, no elegirse después para maximizar un cociente.

Si el presupuesto no permite completar la confirmación, reportar **inconcluso**. Una ampliación requiere una versión nueva del protocolo y sus límites definidos antes de mirar los resultados adicionales; no detener el muestreo justo cuando aparece significación.

### 10.6 Qué se mide y qué contará como patrón reproducido

**Correctitud:** terminales esperados frente a observados; número de scores comparados; diferencias de contadores y scores; decisiones/empates; RF corregido contra CPU y, por separado, RF frente a la verdad; errores, fallos de entrada y cobertura. Para equivalencia CPU/GPU, compartir sólo una fracción de terminales no basta: se exige el conjunto esperado completo.

**Rendimiento:** tiempo total monotónico de alta resolución; CPU-time; memoria RSS y memoria de dispositivo; K no vacíos por trabajador; U, Q, G, B y A; tiempos de parsing, guías, clasificación/colocación, cola residual, subsampling, soporte e I/O. Separar tiempo de cola Slurm del tiempo de cómputo y de la duración calendario con continuaciones.

No sumar tiempo inclusivo de una fase con el de sus propios callees. En ejecución solapada, no sumar tiempos de todos los trabajadores y llamarlos walltime. Registrar la relación entre fases y componentes y señalar intervalos solapados.

**Comparación pareada:** para cada dataset y bloque de hardware, calcular T_CPU_valida/T_candidata sólo si ambas ejecuciones terminaron y pasaron los gates. Resumir las repeticiones técnicas mediante la mediana del log-ratio. Si un dataset se ejecuta en varios bloques, promediar primero esas medianas dentro del dataset. Después promediar los valores de los datasets con igual peso en la escala logarítmica y volver a la escala de speedup con la exponencial. Analizar cada régimen por separado, sin combinar tamaños/modelos distintos como si fueran réplicas de una misma celda. Para corridas encadenadas sin un pareado comparable, reportar tiempos y recursos observacionales, sin atribuirles este contraste confirmatorio.

**Incertidumbre:** intervalo bootstrap percentil del 95 % sobre datasets independientes, 10.000 remuestreos y semilla 41001. Las repeticiones técnicas no cuentan como datasets adicionales. Informar resultados por régimen y backend; no ocultar una regresión mediante una media de celdas favorables. Cuando se mida en varios nodos, mantener el emparejamiento y reportar el efecto de nodo por separado; el intervalo no representa hardware no muestreado.

**Patrón de rendimiento:** una relación entre trabajo efectivo, granularidad y tiempo que se mantenga en datos de confirmación y en el régimen correspondiente. Ajustar primero relaciones por fase y separar N<100 de N>=100; no imponer una complejidad universal ni usar el 82,6 % del perfil pequeño como fracción de producción. Las predicciones para el siguiente escalón se contrastan allí antes de saltar al siguiente.

**Umbral de promoción propuesto:** para la comparación primaria MI210 de extremo a extremo, límite inferior del intervalo de speedup mayor que 1,10, sin violaciones de equivalencia. El 10 % es un umbral práctico de ingeniería, no una observación previa ni una regla para el microkernel. Su alcance queda limitado al régimen confirmado. A100 necesita su propia validación; no trasladar el speedup AMD a NVIDIA.

Si una búsqueda se censura por timeout, registrar la cota temporal y el motivo; no tratar el límite como tiempo de finalización. Una celda confirmatoria con pares incompletos no pasa este criterio de speedup. Nunca eliminar fallos para completar artificialmente diez datasets exitosos. Para correctitud, cualquier discrepancia requiere diagnóstico; un promedio RF pequeño no sustituye la equivalencia pretendida.

Los experimentos biológicos pueden mostrar error incluso en CPU de referencia. Informar esa limitación de señal/modelo; no cambiar silenciosamente los parámetros del simulador para producir una verdad más fácil.

### 10.7 Cambios de implementación y artefactos previstos

Estos archivos/interfaces son objetivos de implementación, no archivos nuevos ya creados por esta planificación:

| Bloque | Repositorio y superficie | Cambio previsto |
|---|---|---|
| C0: RF | genoaligner-devel: scripts/rf_distance.py, tests/test_caster_report.py y workflow CPU | Filtrar ambos lados de la bipartición, ampliar invariancias y ejecutar la suite Python en CI |
| C1: gate numérico | ASTER: validateScores y benchmark del ejecutor | Rechazar no finitos antes de la tolerancia; negativos que prueben que el gate falla |
| C2: identidad y runner | genoaligner-devel: caster_run.sh, caster_pending.py y reportes | Hash de entrada/configuración, reutilización condicionada al manifiesto, duración de alta resolución y estados sin ambigüedad |
| C3: experimentos | genoaligner-devel: nuevo scripts/caster_experiment.py y extensión de caster_scale_report.py | Manifiesto inmutable, celdas/réplicas, invocación de AliSim, consumo de trazas congeladas y análisis pareado; no regenerar verdad al reportar |
| C4: observabilidad | ASTER: algorithms.hpp, sequence.hpp y frontera del ejecutor | Fases y contadores de trabajo lógico, captura de trazas sin duplicarlas por hilo; separar build instrumentado de build cronometrado |
| C5: scheduler/HIP | ASTER: twoStepWorkflow, Tripartition/Gene, ThreadPool y PortableHipExecutor | Estado privado, orden determinista, tiles que no alteran ventanas e integración real de scores |
| C6: recuperación/escala | ASTER y launchers de genoaligner-devel | Estado serializable, checkpoint seguro y luego comparación de una/dos/tres GPU |

Interfaz propuesta del harness:

```text
caster_experiment.py --manifest EXPERIMENT.json --case CASE_ID --out ROOT
```

El manifiesto deberá incluir versión del esquema/protocolo, etapa, identificadores de celda y dataset, generador/versión, árbol/máscara/entrada por hash, parámetros, semillas separadas, configuración CASTER, backends/binarios/compilación, recursos y presupuestos. Definir un variant_id distinto para cada combinación de implementación, flags, hilos, batching y política de ejecución. Cada ejecución añade dimensiones observadas N/L/S/K, hardware/job, hardware_block_id del bloque pareado, repeticiones, resultados y estado.

Artefactos separados por `experiment_id/case_id/data_seed/variant_id/hardware_block_id/rep`; cada tarea escribe su propio resultado de forma atómica. Compartir sólo entradas inmutables entre variantes. No permitir que CPU de 1/4/32 hilos o asignaciones distintas reutilicen el mismo resultado por compartir el nombre de backend. Consolidar posteriormente, sin varios procesos anexando al mismo TSV global. Conservar todos los estados: complete, timeout, failed, invalid_input y not_run; no confundir ausencias con éxitos.

### 10.8 Primera modificación lista para implementar

El primer cambio ejecutable será **C0, RF y sus regresiones**, no otro job grande:

1. Añadir a RfDistanceTest las pruebas de las tres representaciones equivalentes de la sección 5; deben fallar antes de corregir el código.
2. Excluir splits si cualquiera de sus lados tiene cardinalidad menor a dos, tanto en splits como en project, conservando la API de salida.
3. Probar poda a nombres compartidos y permutaciones de etiquetas; mantener el cuarteto diferente con RF=2 y el control estrella frente a cuarteto resuelto con RF=1.
4. Ejecutar e incorporar al gate dedicado el comando:

```bash
python3 -m unittest discover -s tests -p 'test_caster_report.py'
```

5. No regenerar automáticamente los RF históricos. Primero identificar entradas/versiones afectadas; la regeneración será una ejecución separada con procedencia nueva.

Después seguirán C1 y C2, y una celda de E1 para validar la infraestructura experimental. El plan queda listo para comenzar por pruebas que fallen de forma conocida, no por una promesa de aceleración ni por la búsqueda indefinida de un resultado favorable.
