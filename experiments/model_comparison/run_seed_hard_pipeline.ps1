param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(42, 123, 2026)]
    [int]$Seed,

    [ValidateSet('tune', 'system', 'fashionclip', 'openclip', 'siglip', 'all')]
    [string]$Stage = 'all',

    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cpu',

    [ValidateRange(1, 512)]
    [int]$BatchSize = 16
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$projectRoot = Join-Path $repoRoot 'project'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$catalogDir = if ($Seed -eq 42) { 'processed_diverse' } else { "processed_diverse_seed_$Seed" }
$catalog = Join-Path $projectRoot "data\$catalogDir\products.csv"
$runDir = Join-Path $projectRoot "runs\clip_finetune_diverse_seed_$Seed"
$embeddingDir = Join-Path $projectRoot "embeddings_diverse_seed_$Seed"
$metadataDir = Join-Path $projectRoot "metadata_index_diverse_seed_$Seed"
$results = Join-Path $PSScriptRoot "results\diverse_seed_$Seed"
$tuningDir = Join-Path $results 'weight_tuning_hard_v1'
$weightsFile = Join-Path $tuningDir 'selected_runtime_weights.json'
$systemDir = Join-Path $results 'system_pipeline_hard_v1'
$fashionDir = Join-Path $results 'fashionclip_raw_v1'
$openclipDir = Join-Path $results 'openclip_raw_v1'
$siglipDir = Join-Path $results 'siglip_raw_v1'

foreach ($required in @($catalog, (Join-Path $runDir 'training.json'),
        (Join-Path $embeddingDir 'image_embeddings.npy'),
        (Join-Path $embeddingDir 'text_embeddings.npy'), $metadataDir)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing input: $required" }
}

if ($Stage -in @('tune', 'all')) {
    Write-Host "Seed $Seed - selecting retrieval weights on validation" -ForegroundColor Cyan
    Push-Location $projectRoot
    try {
        & $python -m src.tune_retrieval_weights `
            --run-dir $runDir --catalog-csv $catalog `
            --image-embeddings (Join-Path $embeddingDir 'image_embeddings.npy') `
            --text-embeddings (Join-Path $embeddingDir 'text_embeddings.npy') `
            --metadata-index-dir $metadataDir `
            --query-text-cache-dir (Join-Path $projectRoot "cache\validation_seed_$Seed") `
            --expected-gallery-size 5000 --objective-k 10 `
            --device $Device --batch-size $BatchSize `
            --output-dir $tuningDir --weights-output $weightsFile
        if ($LASTEXITCODE -ne 0) { throw "Weight selection failed for seed $Seed" }
    }
    finally { Pop-Location }
}

if ($Stage -in @('system', 'all')) {
    if (-not (Test-Path -LiteralPath $weightsFile)) {
        throw "Selected weights missing: $weightsFile. Run -Stage tune first."
    }
    $weights = (Get-Content -LiteralPath $weightsFile -Raw | ConvertFrom-Json).weights
    Write-Host "Seed $Seed - evaluating deployed Hard Category system pipeline" -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        & $python -m experiments.model_comparison.compare_system_pipeline `
            --run-dir $runDir --catalog-csv $catalog `
            --image-embeddings (Join-Path $embeddingDir 'image_embeddings.npy') `
            --text-embeddings (Join-Path $embeddingDir 'text_embeddings.npy') `
            --metadata-index-dir $metadataDir `
            --query-text-cache-dir (Join-Path $PSScriptRoot "cache\system_seed_$Seed") `
            --expected-gallery-size 5000 --query-split test --ks 1 5 10 `
            --image-weight $weights.COMBINED_IMAGE_WEIGHT `
            --text-image-weight $weights.TEXT_RETRIEVAL_IMAGE_WEIGHT `
            --text-metadata-weight $weights.TEXT_METADATA_WEIGHT `
            --combined-metadata-weight $weights.COMBINED_METADATA_WEIGHT `
            --device $Device --batch-size $BatchSize --output-dir $systemDir
        if ($LASTEXITCODE -ne 0) { throw "System pipeline failed for seed $Seed" }
    }
    finally { Pop-Location }
}

if ($Stage -in @('fashionclip', 'all')) {
    Write-Host "Seed $Seed - evaluating frozen FashionCLIP raw baseline" -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        & $python -m experiments.model_comparison.compare_fashionclip `
            --run-dir $runDir --catalog-csv $catalog `
            --expected-gallery-size 5000 --ks 1 5 10 `
            --device $Device --batch-size $BatchSize `
            --cache-dir (Join-Path $PSScriptRoot "cache\fashionclip_seed_$Seed") `
            --output-dir $fashionDir
        if ($LASTEXITCODE -ne 0) { throw "FashionCLIP baseline failed for seed $Seed" }
    }
    finally { Pop-Location }
}

foreach ($family in @('openclip', 'siglip')) {
    if ($Stage -ne $family) { continue }
    $output = if ($family -eq 'openclip') { $openclipDir } else { $siglipDir }
    Write-Host "Seed $Seed - evaluating frozen $family raw baseline" -ForegroundColor Cyan
    Push-Location $repoRoot
    try {
        & $python -m experiments.model_comparison.compare_additional_clips `
            --model-family $family `
            --run-dir $runDir --catalog-csv $catalog `
            --expected-gallery-size 5000 --ks 1 5 10 `
            --device $Device --batch-size $BatchSize `
            --cache-dir (Join-Path $PSScriptRoot "cache\${family}_seed_$Seed") `
            --output-dir $output
        if ($LASTEXITCODE -ne 0) { throw "$family baseline failed for seed $Seed" }
    }
    finally { Pop-Location }
}

Write-Host "Seed $Seed completed: $Stage" -ForegroundColor Green
