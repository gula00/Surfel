# 在 PowerShell 中运行：.\run.ps1；加 -Validate 可开启同步验证层。
param([switch]$Validate)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { throw 'CMake 配置失败' }
    cmake --build build
    if ($LASTEXITCODE -ne 0) { throw '编译失败' }
    $programArgs = @()
    if ($Validate) { $programArgs += '--validate' }
    & ./build/surfel_vulkan.exe @programArgs
    if ($LASTEXITCODE -ne 0) { throw 'Vulkan 运行或验证失败' }

    # 程序自身只写无依赖的 BMP；Windows 自带组件转成方便预览的 PNG。
    Add-Type -AssemblyName System.Drawing
    foreach ($name in @('direct', 'gi')) {
        $bitmap = [System.Drawing.Bitmap]::new((Join-Path $PSScriptRoot "$name.bmp"))
        try {
            $bitmap.Save((Join-Path $PSScriptRoot "$name.png"), [System.Drawing.Imaging.ImageFormat]::Png)
        } finally { $bitmap.Dispose() }
    }
    Write-Host '完成：direct.png（直接光）、gi.png（直接光 + surfel 间接光）'
} finally { Pop-Location }
