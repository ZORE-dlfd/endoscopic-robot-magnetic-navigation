$ErrorActionPreference = 'Stop'
$paper = Split-Path -Parent $PSScriptRoot
$project = Split-Path -Parent (Split-Path -Parent $paper)
$engine = Join-Path $project '构建\论文工具\tectonic\tectonic.exe'
if (-not (Test-Path -LiteralPath $engine)) {
    throw '找不到独立 Tectonic 编译器；也可用 XeLaTeX 从 source 目录编译主文件。'
}
$cache = 'C:\Users\Public\Documents\CodexTectonicCache'
$temporary = 'C:\Users\Public\Documents\CodexTectonicTemp'
New-Item -ItemType Directory -Path $cache, $temporary -Force | Out-Null
$env:TECTONIC_CACHE_DIR = $cache
$env:TEMP = $temporary
$env:TMP = $temporary
Push-Location (Join-Path $paper 'source')
try {
    & $engine --keep-logs --keep-intermediates --outdir '..\build' 'methodology.tex'
    if ($LASTEXITCODE -ne 0) { throw "LaTeX 编译失败，退出码 $LASTEXITCODE" }
    $log = Get-Content -LiteralPath '..\build\methodology.log' -Raw
    if ($log -match 'Overfull \\hbox|undefined references|undefined citations|Missing character:') {
        throw '编译存在溢出、未解析引用或缺字，需要检查日志后再发布。'
    }
    Copy-Item -LiteralPath '..\build\methodology.pdf' -Destination '..\output\pdf\八线圈磁驱动与进退镜视觉协同方法.pdf' -Force
} finally {
    Pop-Location
}
