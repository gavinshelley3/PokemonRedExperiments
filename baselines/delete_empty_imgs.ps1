Get-ChildItem -Path . -Filter *.jpeg -Recurse | Where-Object { $_.Length -lt 1000 } | Remove-Item