# Synthesise a narration track from the SRT, cue by cue, placed at each cue's own timestamp.
#
# A scratch track, not a finished one: it exists so the cut can be judged with sound and the
# timing checked before anyone books a microphone. Replace it with a real voice before
# publishing -- a synthetic reader on a portfolio piece reads as a shortcut, and this one is
# only here because the alternative was shipping silence and calling it a decision.
param(
  [string]$Srt  = "D:\Projects\warrant\media\warrant-demo.srt",
  [string]$Out  = "D:\Projects\warrant\media\_narration",
  [string]$Voice = "Microsoft Zira Desktop",
  [int]$Rate = 0
)

Add-Type -AssemblyName System.Speech
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Get-ChildItem $Out -Filter *.wav -ErrorAction SilentlyContinue | Remove-Item -Force

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice($Voice)
$synth.Rate = $Rate

# SRT: index / start --> end / text lines / blank
$lines = Get-Content -Path $Srt -Encoding UTF8
$cues = @()
$i = 0
while ($i -lt $lines.Count) {
  if ($lines[$i] -match '^\d+$') {
    $time = $lines[$i + 1]
    if ($time -match '^(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->') {
      $start = [double]$matches[1] * 3600 + [double]$matches[2] * 60 +
               [double]$matches[3] + [double]$matches[4] / 1000
      $j = $i + 2; $text = @()
      while ($j -lt $lines.Count -and $lines[$j].Trim() -ne '') { $text += $lines[$j]; $j++ }
      $cues += [pscustomobject]@{ Start = $start; Text = ($text -join ' ') }
      $i = $j
    }
  }
  $i++
}

$n = 0
foreach ($c in $cues) {
  $n++
  $file = Join-Path $Out ("cue_{0:d3}_{1}.wav" -f $n, [int]($c.Start * 1000))
  $synth.SetOutputToWaveFile($file)
  $synth.Speak($c.Text)
  $synth.SetOutputToNull()
}
$synth.Dispose()
"  wrote $n cues to $Out"
