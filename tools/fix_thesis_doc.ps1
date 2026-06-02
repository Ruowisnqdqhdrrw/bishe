param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [string]$BackupPath
)

$ErrorActionPreference = 'Stop'

$script:TableChar = [string][char]0x8868
$script:ContinuationSuffix = ([string][char]0xFF08) + ([string][char]0x7EED) + ([string][char]0xFF09)

function Clean-Text([string]$Text) {
    if ($null -eq $Text) {
        return ''
    }

    return ($Text -replace '[\r\a\v\f]+', ' ' -replace '\s+', ' ').Trim()
}

function Get-PageNumber($Document, [int]$Position) {
    $safe = [Math]::Max(0, $Position)
    return $Document.Range($safe, $safe).Information(3)
}

function Get-RangeEndPage($Document, $Range) {
    $endPos = [Math]::Max($Range.Start, $Range.End - 1)
    return Get-PageNumber $Document $endPos
}

function Get-PreviousNonEmptyParagraph($Document, [int]$Position) {
    $range = $Document.Range(0, [Math]::Max(0, $Position))
    $paragraphs = $range.Paragraphs
    for ($i = $paragraphs.Count; $i -ge 1; $i--) {
        $paragraph = $paragraphs.Item($i)
        if ((Clean-Text $paragraph.Range.Text) -ne '') {
            return $paragraph
        }
    }

    return $null
}

function Get-ContinuationCaption([string]$CaptionText) {
    $pattern = '^(' + [regex]::Escape($script:TableChar) + '\s*\d+\.\d+)'
    if ($CaptionText -match $pattern) {
        return ($matches[1] + $script:ContinuationSuffix)
    }

    return ($CaptionText + $script:ContinuationSuffix)
}

function Set-HeaderBottomBorder($Document) {
    for ($sectionIndex = 1; $sectionIndex -le $Document.Sections.Count; $sectionIndex++) {
        $section = $Document.Sections.Item($sectionIndex)
        $header = $section.Headers.Item(1)
        if ((Clean-Text $header.Range.Text) -eq '') {
            continue
        }

        $paragraph = $header.Range.Paragraphs.Item(1)
        $bottomBorder = $paragraph.Borders.Item(-3)
        $bottomBorder.LineStyle = 1
        $bottomBorder.LineWidth = 4
        $bottomBorder.Color = 0
    }
}

function Set-TableColumnWidths($Table, [double[]]$Widths) {
    $sum = ($Widths | Measure-Object -Sum).Sum
    $Table.AllowAutoFit = $false
    $Table.PreferredWidthType = 3
    $Table.PreferredWidth = $sum
    $Table.Rows.LeftIndent = 0

    for ($columnIndex = 1; $columnIndex -le $Widths.Count; $columnIndex++) {
        $Table.Columns.Item($columnIndex).Width = $Widths[$columnIndex - 1]
    }
}

function Set-TableParagraphAlignment($Table, [int]$Alignment) {
    for ($rowIndex = 1; $rowIndex -le $Table.Rows.Count; $rowIndex++) {
        for ($columnIndex = 1; $columnIndex -le $Table.Columns.Count; $columnIndex++) {
            $cell = $Table.Cell($rowIndex, $columnIndex)
            for ($paragraphIndex = 1; $paragraphIndex -le $cell.Range.Paragraphs.Count; $paragraphIndex++) {
                $cell.Range.Paragraphs.Item($paragraphIndex).Alignment = $Alignment
            }
        }
    }
}

function Set-RowPaginationRules($Table) {
    if ($Table.Rows.Count -ge 1) {
        $Table.Rows.Item(1).HeadingFormat = -1
    }

    for ($rowIndex = 1; $rowIndex -le $Table.Rows.Count; $rowIndex++) {
        $Table.Rows.Item($rowIndex).AllowBreakAcrossPages = 0
    }
}

function Copy-HeaderRowToTop($SourceTable, $TargetTable, $WordApp) {
    $TargetTable.Rows.Item(1).Select() | Out-Null
    $WordApp.Selection.InsertRowsAbove(1) | Out-Null

    for ($columnIndex = 1; $columnIndex -le $TargetTable.Columns.Count; $columnIndex++) {
        try {
            $TargetTable.Cell(1, $columnIndex).Range.FormattedText = $SourceTable.Cell(1, $columnIndex).Range.FormattedText
        }
        catch {
            $TargetTable.Cell(1, $columnIndex).Range.Text = Clean-Text $SourceTable.Cell(1, $columnIndex).Range.Text
            $TargetTable.Cell(1, $columnIndex).Range.Paragraphs.Item(1).Alignment = 1
        }
    }

    $TargetTable.Rows.Item(1).HeadingFormat = -1
}

function Insert-ContinuationCaption($Document, $Table, $ReferenceParagraph, [string]$CaptionText) {
    $insertAt = $Table.Range.Start
    $Document.Range($insertAt, $insertAt).InsertBefore($CaptionText + "`r")

    $paragraph = Get-PreviousNonEmptyParagraph $Document $Table.Range.Start
    if ($null -eq $paragraph) {
        return
    }

    try {
        $paragraph.Range.Style = $ReferenceParagraph.Range.Style
    }
    catch {
    }

    $paragraph.Alignment = $ReferenceParagraph.Alignment
    $paragraph.SpaceBefore = $ReferenceParagraph.SpaceBefore
    $paragraph.SpaceAfter = $ReferenceParagraph.SpaceAfter
    $paragraph.KeepTogether = 0
    $paragraph.KeepWithNext = 0
    $paragraph.PageBreakBefore = 0
}

function Split-TableAcrossPages($Document, $WordApp, $Table, [string]$CaptionText) {
    $referenceParagraph = Get-PreviousNonEmptyParagraph $Document $Table.Range.Start
    if ($null -eq $referenceParagraph) {
        return
    }

    $currentTable = $Table
    while ((Get-RangeEndPage $Document $currentTable.Range) -gt (Get-PageNumber $Document $currentTable.Range.Start)) {
        $startPage = Get-PageNumber $Document $currentTable.Range.Start
        $splitRowIndex = $null

        for ($rowIndex = 2; $rowIndex -le $currentTable.Rows.Count; $rowIndex++) {
            $rowStartPage = Get-PageNumber $Document $currentTable.Rows.Item($rowIndex).Range.Start
            if ($rowStartPage -gt $startPage) {
                $splitRowIndex = $rowIndex
                break
            }
        }

        if ($null -eq $splitRowIndex) {
            for ($rowIndex = 2; $rowIndex -le $currentTable.Rows.Count; $rowIndex++) {
                $row = $currentTable.Rows.Item($rowIndex)
                if ((Get-RangeEndPage $Document $row.Range) -gt $startPage) {
                    $splitRowIndex = $rowIndex
                    break
                }
            }
        }

        if ($null -eq $splitRowIndex) {
            break
        }

        $currentIndex = $currentTable.Index
        $currentTable.Rows.Item($splitRowIndex).Range.Select() | Out-Null
        $WordApp.Selection.SplitTable() | Out-Null

        $nextTable = $Document.Tables.Item($currentIndex + 1)
        Insert-ContinuationCaption $Document $nextTable $referenceParagraph (Get-ContinuationCaption $CaptionText)
        Copy-HeaderRowToTop $currentTable $nextTable $WordApp
        Set-RowPaginationRules $currentTable
        Set-RowPaginationRules $nextTable

        $currentTable = $nextTable
        $Document.Repaginate() | Out-Null
    }
}

if ($BackupPath) {
    Copy-Item -LiteralPath $Path -Destination $BackupPath -Force
}

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.ScreenUpdating = $false

    $document = $word.Documents.Open($Path, $false, $true)

    Set-HeaderBottomBorder $document

    $tableTargets = @(
        @{ Regex = ('^' + [regex]::Escape($script:TableChar) + '\s*4\.2\b'); Widths = @(105, 140, 95, 137) },
        @{ Regex = ('^' + [regex]::Escape($script:TableChar) + '\s*5\.4\b'); Widths = @(78, 105, 150, 88, 56) }
    )

    $tableMetadata = @()
    for ($tableIndex = 1; $tableIndex -le $document.Tables.Count; $tableIndex++) {
        $table = $document.Tables.Item($tableIndex)
        $captionParagraph = Get-PreviousNonEmptyParagraph $document $table.Range.Start
        $captionText = if ($null -ne $captionParagraph) { Clean-Text $captionParagraph.Range.Text } else { '' }

        foreach ($target in $tableTargets) {
            if ($captionText -match $target.Regex) {
                Set-TableColumnWidths $table $target.Widths
                Set-TableParagraphAlignment $table 1
            }
        }

        Set-RowPaginationRules $table
        $tableMetadata += [pscustomobject]@{
            Index = $tableIndex
            Caption = $captionText
        }
    }

    $document.Repaginate() | Out-Null

    for ($metaIndex = $tableMetadata.Count - 1; $metaIndex -ge 0; $metaIndex--) {
        $meta = $tableMetadata[$metaIndex]
        $table = $document.Tables.Item($meta.Index)
        if ((Get-RangeEndPage $document $table.Range) -gt (Get-PageNumber $document $table.Range.Start)) {
            Split-TableAcrossPages $document $word $table $meta.Caption
        }
    }

    $document.Repaginate() | Out-Null
    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    $document.SaveAs([ref]$OutputPath)
}
finally {
    if ($null -ne $document) {
        $document.Close() | Out-Null
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($document) | Out-Null
    }

    if ($null -ne $word) {
        $word.Quit() | Out-Null
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    }

    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
