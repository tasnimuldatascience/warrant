# Record the terminal demo. Opens a maximised console running scripts/demo.py, captures the
# desktop with ffmpeg for the duration, and closes it.
param(
  [int]$Seconds = 78,
  [string]$Out = "D:\Projects\warrant\media\warrant-demo-raw.mp4"
)

$ff = "C:\Users\hrido\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
$repo = "D:\Projects\warrant"

# A dark, large-font console so the capture is legible at 1080p after scaling.
$inner = @"
`$Host.UI.RawUI.WindowTitle = 'warrant'
Set-Location '$repo'
Clear-Host
python scripts/demo.py --pace 0.55
Start-Sleep -Seconds 4
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($inner))

$proc = Start-Process powershell.exe -PassThru -WindowStyle Maximized `
  -ArgumentList '-NoProfile', '-NoExit', '-EncodedCommand', $encoded
Start-Sleep -Seconds 3

& $ff -y -f gdigrab -framerate 12 -t $Seconds -i desktop `
     -vf "scale=1920:-2:flags=lanczos" -c:v libx264 -preset veryfast -crf 20 `
     -pix_fmt yuv420p $Out 2>&1 | Select-Object -Last 2

Start-Sleep -Seconds 1
if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
"recorded -> $Out"
