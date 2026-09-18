$ErrorActionPreference = 'Stop'

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $jobs = @(
        @{
            Input = (Resolve-Path -LiteralPath 'papers\PaperID 804 final.docx').Path
            Output = (Join-Path (Resolve-Path -LiteralPath 'papers\.review_revision_work').Path 'PaperID_804_final_render.pdf')
        },
        @{
            Input = (Resolve-Path -LiteralPath 'papers\Response_to_Reviewers_final.docx').Path
            Output = (Join-Path (Resolve-Path -LiteralPath 'papers\.review_revision_work').Path 'Response_to_Reviewers_final_render.pdf')
        }
    )

    foreach ($job in $jobs) {
        $doc = $word.Documents.Open($job.Input, $false, $true)
        try {
            $doc.Repaginate()
            $pages = $doc.ComputeStatistics(2)
            $doc.ExportAsFixedFormat($job.Output, 17)
            Write-Output "$($job.Input)`t$pages`t$($job.Output)"
        }
        finally {
            $doc.Close(0)
        }
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
}
