$ErrorActionPreference = 'Stop'

$manuscriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TEXINPUTS = "$manuscriptDirectory;"
$env:TEXFONTMAPS = "$manuscriptDirectory;"
$env:T1FONTS = "$manuscriptDirectory;"
$env:TFMFONTS = "$manuscriptDirectory;"

Push-Location $manuscriptDirectory
try {
    & pdflatex -interaction=nonstopmode -halt-on-error ACCESS_ConceptDrift_Cyber.tex
    if ($LASTEXITCODE -ne 0) { throw 'First pdflatex pass failed.' }
    & bibtex ACCESS_ConceptDrift_Cyber
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    & pdflatex -interaction=nonstopmode -halt-on-error ACCESS_ConceptDrift_Cyber.tex
    if ($LASTEXITCODE -ne 0) { throw 'Second pdflatex pass failed.' }
    & pdflatex -interaction=nonstopmode -halt-on-error ACCESS_ConceptDrift_Cyber.tex
    if ($LASTEXITCODE -ne 0) { throw 'Final pdflatex pass failed.' }
}
finally {
    Pop-Location
}

Write-Output "Built $manuscriptDirectory\ACCESS_ConceptDrift_Cyber.pdf"
