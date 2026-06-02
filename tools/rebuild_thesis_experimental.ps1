param(
    [string]$SourcePath = "D:\计算机本科毕业论文\论文极限格式规范版.docx",
    [string]$TemplatePath = "D:\计算机本科毕业论文\附件1：江苏大学毕业设计（论文）通用格式模板（理工农医教）-2026版.doc",
    [string]$OutputDocx = "D:\计算机本科毕业论文\论文终极实验版.docx",
    [string]$OutputDoc = "D:\计算机本科毕业论文\论文终极实验版.doc"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdAlignParagraphRight = 2
$wdAlignParagraphJustify = 3
$wdAlignTabRight = 2
$wdLineSpace1pt5 = 1
$wdHeaderFooterPrimary = 1
$wdAlignPageNumberCenter = 1
$wdPageNumberStyleArabic = 0
$wdPageNumberStyleUppercaseRoman = 1
$wdTabLeaderDots = 1
$wdCollapseEnd = 0
$wdSectionBreakNextPage = 2
$wdFormatDocumentDefault = 16
$wdFormatDocument97 = 0
$wdWithInTable = 12
$wdInfoPage = 3

function Get-ParaText([object]$para) {
    return (($para.Range.Text -replace "[`r`a]", "")).Trim()
}

function Get-CellText([object]$cell) {
    return (($cell.Range.Text -replace "[`r`a]", "")).Trim()
}

function Get-RangeText([object]$range) {
    return ($range.Text -replace "[`r`a]", "")
}

function Find-ParagraphIndexByExactText(
    [object]$doc,
    [string]$text,
    [int]$startIndex = 1
) {
    for ($i = $startIndex; $i -le $doc.Paragraphs.Count; $i++) {
        if ((Get-ParaText $doc.Paragraphs.Item($i)) -eq $text) {
            return $i
        }
    }
    throw "Paragraph not found: $text"
}

function Find-ParagraphIndexByRegex(
    [object]$doc,
    [string]$pattern,
    [int]$startIndex = 1
) {
    for ($i = $startIndex; $i -le $doc.Paragraphs.Count; $i++) {
        if ((Get-ParaText $doc.Paragraphs.Item($i)) -match $pattern) {
            return $i
        }
    }
    throw "Paragraph pattern not found: $pattern"
}

function Find-ParagraphIndexByStyleAndRegex(
    [object]$doc,
    [string]$styleName,
    [string]$pattern,
    [int]$startIndex = 1
) {
    for ($i = $startIndex; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $text = Get-ParaText $para
        if (-not $text) {
            continue
        }
        $currentStyle = ""
        try {
            $currentStyle = $para.Range.Style.NameLocal
        } catch {
            $currentStyle = ""
        }
        if ($currentStyle -eq $styleName -and $text -match $pattern) {
            return $i
        }
    }
    throw "Paragraph not found for style [$styleName]: $pattern"
}

function Get-RangeBetweenParagraphs(
    [object]$doc,
    [int]$startIndex,
    [int]$endIndexExclusive
) {
    $start = $doc.Paragraphs.Item($startIndex).Range.Start
    $end = if ($endIndexExclusive -le $doc.Paragraphs.Count) {
        $doc.Paragraphs.Item($endIndexExclusive).Range.Start
    } else {
        $doc.Content.End - 1
    }
    return $doc.Range($start, $end)
}

function Get-TrimmedRangeBetweenParagraphs(
    [object]$doc,
    [int]$startIndex,
    [int]$endIndexExclusive
) {
    $trimStart = $startIndex
    while ($trimStart -lt $endIndexExclusive -and -not (Get-ParaText $doc.Paragraphs.Item($trimStart))) {
        $trimStart++
    }

    $trimLast = $endIndexExclusive - 1
    while ($trimLast -ge $trimStart -and -not (Get-ParaText $doc.Paragraphs.Item($trimLast))) {
        $trimLast--
    }

    if ($trimLast -lt $trimStart) {
        return $doc.Range($doc.Paragraphs.Item($startIndex).Range.Start, $doc.Paragraphs.Item($startIndex).Range.Start)
    }
    return Get-RangeBetweenParagraphs $doc $trimStart ($trimLast + 1)
}

function Get-EditableSectionRange([object]$doc, [int]$sectionIndex) {
    $section = $doc.Sections.Item($sectionIndex)
    return $doc.Range($section.Range.Start, [Math]::Max($section.Range.Start, $section.Range.End - 1))
}

function Get-TocEditableRange([object]$doc) {
    $section = $doc.Sections.Item(5)
    $paragraphs = $section.Range.Paragraphs
    $end = [Math]::Max($section.Range.Start, $section.Range.End - 1)
    if ($paragraphs.Count -ge 2) {
        $end = [Math]::Max($section.Range.Start, $paragraphs.Item($paragraphs.Count - 1).Range.Start)
    }
    return $doc.Range($section.Range.Start, $end)
}

function Get-SectionParagraphs([object]$doc, [int]$sectionIndex) {
    $section = $doc.Sections.Item($sectionIndex)
    $start = $section.Range.Start
    $end = $section.Range.End
    $items = New-Object System.Collections.ArrayList
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -ge $start -and $para.Range.Start -lt $end) {
            [void]$items.Add($para)
        }
    }
    return $items
}

function Set-RangeFont(
    [object]$range,
    [string]$farEast,
    [string]$ascii,
    [double]$size,
    [bool]$bold
) {
    $range.Font.NameFarEast = $farEast
    $range.Font.NameAscii = $ascii
    $range.Font.NameOther = $ascii
    $range.Font.Name = $ascii
    $range.Font.Size = $size
    $range.Font.Bold = $bold
    $range.Font.Color = -16777216
    $range.Font.Underline = 0
}

function Set-ParagraphFormat(
    [object]$para,
    [string]$farEast,
    [string]$ascii,
    [double]$size,
    [bool]$bold,
    [int]$alignment,
    [double]$spaceBefore,
    [double]$spaceAfter,
    [double]$firstLineIndent
) {
    $content = $para.Range.Duplicate
    $content.End = [Math]::Max($content.Start, $content.End - 1)
    Set-RangeFont $content $farEast $ascii $size $bold
    $para.Alignment = $alignment
    $para.LineSpacingRule = $wdLineSpace1pt5
    $para.SpaceBefore = $spaceBefore
    $para.SpaceAfter = $spaceAfter
    $para.FirstLineIndent = $firstLineIndent
    $para.LeftIndent = 0
    $para.CharacterUnitLeftIndent = 0
    $para.CharacterUnitFirstLineIndent = 0
}

function Set-FooterPageNumbers([object]$section, [int]$numberStyle, [bool]$restart, [int]$startingNumber) {
    $footer = $section.Footers.Item($wdHeaderFooterPrimary)
    $footer.LinkToPrevious = $false
    $footer.Range.Text = ""
    $footer.Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    Set-RangeFont $footer.Range "宋体" "Times New Roman" 9 $false
    $pageNumbers = $footer.PageNumbers
    while ($pageNumbers.Count -gt 0) {
        $pageNumbers.Item(1).Delete()
    }
    [void]$pageNumbers.Add($wdAlignPageNumberCenter, $true)
    $pageNumbers.NumberStyle = $numberStyle
    $pageNumbers.RestartNumberingAtSection = $restart
    if ($restart) {
        $pageNumbers.StartingNumber = $startingNumber
    }
}

function Set-ParagraphTextPreserveMark([object]$para, [string]$text) {
    $content = $para.Range.Duplicate
    $content.End = [Math]::Max($content.Start, $content.End - 1)
    $content.Text = $text
}

function Remove-ExistingBookmarkIfPresent([object]$doc, [string]$bookmarkName) {
    if ($doc.Bookmarks.Exists($bookmarkName)) {
        $doc.Bookmarks.Item($bookmarkName).Delete()
    }
}

function Clear-ParagraphShading([object]$para) {
    try {
        $para.Shading.BackgroundPatternColor = 16777215
        $para.Shading.ForegroundPatternColor = 16777215
        $para.Shading.Texture = 0
    } catch {
    }
    try {
        $para.Range.Shading.BackgroundPatternColor = 16777215
        $para.Range.Shading.ForegroundPatternColor = 16777215
        $para.Range.Shading.Texture = 0
    } catch {
    }
    try {
        $para.Range.HighlightColorIndex = 0
    } catch {
    }
}

function Get-CoverMetadata([object]$sourceDoc) {
    $declarationIndex = Find-ParagraphIndexByExactText $sourceDoc "毕业设计（论文）原创性声明"
    $texts = New-Object System.Collections.Generic.List[string]
    for ($i = 1; $i -lt $declarationIndex; $i++) {
        $text = Get-ParaText $sourceDoc.Paragraphs.Item($i)
        if (-not $text) {
            continue
        }
        if ($text -match '^(学院名称|专业班级|学生姓名|学生学号|指导教师姓名|指导教师职称)') {
            continue
        }
        if ($text -match 'J\s*I\s*N\s*G\s*S\s*U') {
            continue
        }
        if ($text -eq '本科  毕  业  设  计（论  文）' -or $text -eq '本  科  毕  业  设  计（论  文）') {
            continue
        }
        [void]$texts.Add($text)
    }

    $cnTitle = ""
    $enTitle = ""
    $monthText = ""
    foreach ($text in $texts) {
        if (-not $cnTitle -and $text -match '[一-龥]' -and $text.Length -ge 8 -and $text -notmatch '年\d+月') {
            $cnTitle = $text
            continue
        }
        if (-not $enTitle -and $cnTitle -and $text -match '[A-Za-z]' -and $text.Length -ge 10) {
            $enTitle = $text
            continue
        }
        if ($text -match '^\d{4}年\d{1,2}月$') {
            $monthText = $text
        }
    }
    if (-not $cnTitle) { throw "Unable to extract Chinese title from source cover." }
    if (-not $enTitle) { throw "Unable to extract English title from source cover." }
    if (-not $monthText) { $monthText = "2026年6月" }

    $values = @()
    $sourceCoverTable = $null
    for ($t = 1; $t -le $sourceDoc.Tables.Count; $t++) {
        $table = $sourceDoc.Tables.Item($t)
        if ($table.Range.Start -lt $sourceDoc.Sections.Item(2).Range.Start) {
            $sourceCoverTable = $table
            break
        }
    }
    if ($sourceCoverTable -ne $null) {
        for ($row = 1; $row -le [Math]::Min(6, $sourceCoverTable.Rows.Count); $row++) {
            $values += (Get-CellText $sourceCoverTable.Cell($row, 2))
        }
    }

    return @{
        CnTitle = $cnTitle
        EnTitle = $enTitle
        MonthText = $monthText
        CoverValues = $values
        DeclarationIndex = $declarationIndex
    }
}

function Fill-Cover([object]$doc, [hashtable]$coverInfo) {
    foreach ($headerType in 1..3) {
        try { $doc.Sections.Item(1).Headers.Item($headerType).Range.Text = "" } catch {}
    }
    foreach ($para in (Get-SectionParagraphs $doc 1)) {
        $text = Get-ParaText $para
        if ($text -like '中文论文题目*') {
            Set-ParagraphTextPreserveMark $para $coverInfo.CnTitle
            Set-ParagraphFormat $para "黑体" "黑体" 22 $false $wdAlignParagraphCenter 0 0 0
        } elseif ($text -like '英文论文题目*') {
            Set-ParagraphTextPreserveMark $para $coverInfo.EnTitle
            Set-ParagraphFormat $para "宋体" "Times New Roman" 16 $true $wdAlignParagraphCenter 0 0 0
        } elseif ($text -match '^年\s*月' -or $text -like '*小四宋体*') {
            if ($text -match '年' -and $text -match '月') {
                Set-ParagraphTextPreserveMark $para $coverInfo.MonthText
                Set-ParagraphFormat $para "宋体" "Times New Roman" 12 $false $wdAlignParagraphCenter 0 0 0
            }
        }
    }

    $coverTable = $doc.Tables.Item(1)
    for ($row = 1; $row -le [Math]::Min(6, $coverTable.Rows.Count); $row++) {
        $value = ""
        if ($row -le $coverInfo.CoverValues.Count) {
            $value = $coverInfo.CoverValues[$row - 1]
        }
        $coverTable.Cell($row, 2).Range.Text = $value
    }
}

function Tighten-CoverLayout([object]$doc, [string]$monthText) {
    for ($pass = 0; $pass -lt 6; $pass++) {
        $monthPara = $null
        $blankCandidate = $null
        $sectionParagraphs = Get-SectionParagraphs $doc 1
        foreach ($para in $sectionParagraphs) {
            if ((Get-ParaText $para) -eq $monthText) {
                $monthPara = $para
                break
            }
        }
        if ($monthPara -eq $null) {
            break
        }
        if ($monthPara.Range.Information(3) -le 1) {
            break
        }

        for ($i = $sectionParagraphs.Count - 1; $i -ge 0; $i--) {
            $para = $sectionParagraphs[$i]
            if ($para.Range.Start -ge $monthPara.Range.Start) {
                continue
            }
            if ($para.Range.Information($wdWithInTable)) {
                continue
            }
            if (-not (Get-ParaText $para)) {
                $blankCandidate = $para
                break
            }
        }
        if ($blankCandidate -eq $null) {
            break
        }
        [void]$blankCandidate.Range.Delete()
    }
}

function Remove-CoverMarker([object]$doc) {
    for ($i = $doc.Paragraphs.Count; $i -ge 1; $i--) {
        $para = $doc.Paragraphs.Item($i)
        $text = Get-ParaText $para
        $styleName = ""
        try { $styleName = $para.Range.Style.NameLocal } catch {}
        if ($text -eq "附件1" -and $styleName -eq "页眉") {
            [void]$para.Range.Delete()
            break
        }
    }
}

function Replace-SectionContent([object]$doc, [int]$sectionIndex, [object]$sourceRange) {
    $targetRange = Get-EditableSectionRange $doc $sectionIndex
    $targetRange.FormattedText = $sourceRange.FormattedText
}

function Configure-Headers([object]$doc) {
    $usableWidth = $doc.PageSetup.PageWidth - $doc.PageSetup.LeftMargin - $doc.PageSetup.RightMargin
    $labels = @{
        2 = "原创性声明"
        3 = "摘要"
        4 = "ABSTRACT"
        5 = "目录"
    }

    for ($s = 1; $s -le $doc.Sections.Count; $s++) {
        $header = $doc.Sections.Item($s).Headers.Item($wdHeaderFooterPrimary)
        $header.LinkToPrevious = $false
        $header.Range.Text = ""
        $header.Range.ParagraphFormat.TabStops.ClearAll()
        $header.Range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    }

    $doc.Sections.Item(1).Headers.Item($wdHeaderFooterPrimary).Range.Text = ""

    foreach ($sectionIndex in 2..[Math]::Min(5, $doc.Sections.Count)) {
        $header = $doc.Sections.Item($sectionIndex).Headers.Item($wdHeaderFooterPrimary)
        $header.Range.Text = "$($labels[$sectionIndex])`t江苏大学本科毕业设计（论文）"
        $header.Range.ParagraphFormat.TabStops.ClearAll()
        [void]$header.Range.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, 0)
        $header.Range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
        Set-RangeFont $header.Range "宋体" "Times New Roman" 9 $false
    }

    if ($doc.Sections.Count -ge 6) {
        $bodyHeader = $doc.Sections.Item(6).Headers.Item($wdHeaderFooterPrimary)
        $bodyHeader.Range.Text = "江苏大学本科毕业设计（论文）"
        $bodyHeader.Range.ParagraphFormat.TabStops.ClearAll()
        $bodyHeader.Range.ParagraphFormat.Alignment = $wdAlignParagraphRight
        Set-RangeFont $bodyHeader.Range "宋体" "Times New Roman" 9 $false
    }
}

function Configure-PageNumbers([object]$doc) {
    for ($s = 1; $s -le $doc.Sections.Count; $s++) {
        $doc.Sections.Item($s).Headers.Item($wdHeaderFooterPrimary).LinkToPrevious = $false
        $doc.Sections.Item($s).Footers.Item($wdHeaderFooterPrimary).LinkToPrevious = $false
    }
    $doc.Sections.Item(1).Footers.Item($wdHeaderFooterPrimary).Range.Text = ""
    Set-FooterPageNumbers $doc.Sections.Item(2) $wdPageNumberStyleUppercaseRoman $true 1
    Set-FooterPageNumbers $doc.Sections.Item(3) $wdPageNumberStyleUppercaseRoman $false 1
    Set-FooterPageNumbers $doc.Sections.Item(4) $wdPageNumberStyleUppercaseRoman $false 1
    Set-FooterPageNumbers $doc.Sections.Item(5) $wdPageNumberStyleUppercaseRoman $false 1
    Set-FooterPageNumbers $doc.Sections.Item(6) $wdPageNumberStyleArabic $true 1
}

function Ensure-TocBodySectionBreak([object]$doc) {
    if ($doc.Sections.Count -ge 6) {
        $tocSectionEnd = $doc.Sections.Item(5).Range.End
        $bodySectionStart = $doc.Sections.Item(6).Range.Start
        if ($bodySectionStart -eq $tocSectionEnd) {
            return
        }
    }

    $tocHeadingIndex = Find-ParagraphIndexByRegex $doc '^目\s*录$'
    $bodyHeadingIndex = Find-ParagraphIndexByStyleAndRegex $doc "标题 1" '^\d+\s+' ($tocHeadingIndex + 1)
    $bodyPara = $doc.Paragraphs.Item($bodyHeadingIndex)
    if ($bodyPara.Range.Information(2) -eq 5) {
        $insertRange = $doc.Range($bodyPara.Range.Start, $bodyPara.Range.Start)
        $insertRange.InsertBreak($wdSectionBreakNextPage)
    }
}

function Normalize-FrontMatter([object]$doc) {
    foreach ($sectionIndex in 2..5) {
        foreach ($para in (Get-SectionParagraphs $doc $sectionIndex)) {
            try { $para.PageBreakBefore = $false } catch {}
        }
    }
}

function Format-ParagraphWithPrefix(
    [object]$para,
    [int]$prefixLength,
    [string]$bodyFarEast,
    [string]$bodyAscii,
    [double]$bodySize,
    [bool]$bodyBold,
    [string]$prefixFarEast,
    [string]$prefixAscii,
    [double]$prefixSize,
    [bool]$prefixBold,
    [int]$alignment,
    [double]$spaceBefore,
    [double]$spaceAfter,
    [double]$firstLineIndent
) {
    $content = $para.Range.Duplicate
    $content.End = [Math]::Max($content.Start, $content.End - 1)
    Set-RangeFont $content $bodyFarEast $bodyAscii $bodySize $bodyBold
    if (($content.End - $content.Start) -ge $prefixLength) {
        $prefix = $content.Duplicate
        $prefix.End = $prefix.Start + $prefixLength
        Set-RangeFont $prefix $prefixFarEast $prefixAscii $prefixSize $prefixBold
    }
    $para.Alignment = $alignment
    $para.LineSpacingRule = $wdLineSpace1pt5
    $para.SpaceBefore = $spaceBefore
    $para.SpaceAfter = $spaceAfter
    $para.FirstLineIndent = $firstLineIndent
    $para.LeftIndent = 0
    $para.CharacterUnitLeftIndent = 0
}

function Apply-FrontMatterFormatting(
    [object]$doc,
    [string]$thesisCnTitle,
    [string]$thesisEnTitle
) {
    $statementTitleIndex = Find-ParagraphIndexByExactText $doc "毕业设计（论文）原创性声明"
    $abstractCnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisCnTitle ($statementTitleIndex + 1)
    $abstractEnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisEnTitle ($abstractCnTitleIndex + 1)
    $tocHeadingIndex = Find-ParagraphIndexByRegex $doc '^目\s*录$' ($abstractEnTitleIndex + 1)

    Set-ParagraphFormat $doc.Paragraphs.Item($statementTitleIndex) "黑体" "黑体" 16 $false $wdAlignParagraphCenter 31.2 0 0
    for ($i = $statementTitleIndex + 1; $i -lt $abstractCnTitleIndex; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $text = Get-ParaText $para
        if (-not $text) {
            continue
        }
        Set-ParagraphFormat $para "宋体" "Times New Roman" 14 $false $wdAlignParagraphJustify 0 0 28
        if ($text -like '论文作者签名*') {
            $para.FirstLineIndent = 0
            $para.CharacterUnitFirstLineIndent = 0
            $para.FirstLineIndent = 270
        }
        if ($text -like '日期：*') {
            $para.CharacterUnitFirstLineIndent = 0
            $para.FirstLineIndent = 0
        }
    }

    Set-ParagraphFormat $doc.Paragraphs.Item($abstractCnTitleIndex) "黑体" "黑体" 16 $false $wdAlignParagraphCenter 15.6 7.8 0
    if ($abstractCnTitleIndex + 1 -lt $doc.Paragraphs.Count) {
        Set-ParagraphFormat $doc.Paragraphs.Item($abstractCnTitleIndex + 1) "宋体" "Times New Roman" 12 $false $wdAlignParagraphCenter 0 0 0
    }
    if ($abstractCnTitleIndex + 2 -lt $doc.Paragraphs.Count) {
        Set-ParagraphFormat $doc.Paragraphs.Item($abstractCnTitleIndex + 2) "宋体" "Times New Roman" 12 $false $wdAlignParagraphCenter 0 0 0
    }
    if ($abstractCnTitleIndex + 3 -lt $doc.Paragraphs.Count) {
        Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractCnTitleIndex + 3) 3 "宋体" "Times New Roman" 12 $false "黑体" "黑体" 14 $false $wdAlignParagraphJustify 15.6 0 28
    }
    if ($abstractCnTitleIndex + 4 -lt $doc.Paragraphs.Count) {
        Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractCnTitleIndex + 4) 4 "宋体" "Times New Roman" 12 $false "黑体" "黑体" 14 $false $wdAlignParagraphLeft 15.6 0 0
    }

    Set-ParagraphFormat $doc.Paragraphs.Item($abstractEnTitleIndex) "黑体" "Times New Roman" 14 $true $wdAlignParagraphCenter 15.6 15.6 0
    if ($abstractEnTitleIndex + 1 -lt $doc.Paragraphs.Count) {
        Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractEnTitleIndex + 1) 8 "宋体" "Times New Roman" 12 $false "宋体" "Times New Roman" 12 $true $wdAlignParagraphJustify 0 0 12
    }
    if ($abstractEnTitleIndex + 2 -lt $doc.Paragraphs.Count) {
        Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractEnTitleIndex + 2) 10 "宋体" "Times New Roman" 12 $false "宋体" "Times New Roman" 12 $true $wdAlignParagraphLeft 15.6 0 0
    }

    Set-ParagraphFormat $doc.Paragraphs.Item($tocHeadingIndex) "黑体" "Times New Roman" 22 $false $wdAlignParagraphCenter 15.6 7.8 0
}

function Normalize-BodyHeadings([object]$doc) {
    $firstHeading1Seen = $false
    foreach ($para in (Get-SectionParagraphs $doc 6)) {
        $styleName = ""
        try { $styleName = $para.Range.Style.NameLocal } catch {}
        switch ($styleName) {
            "标题 1" {
                Set-ParagraphFormat $para "黑体" "Times New Roman" 18 $true $wdAlignParagraphCenter 31.2 15.6 0
                $para.PageBreakBefore = $firstHeading1Seen
                $firstHeading1Seen = $true
            }
            "标题 2" {
                Set-ParagraphFormat $para "宋体" "Times New Roman" 14 $true $wdAlignParagraphLeft 7.8 7.8 0
                $para.PageBreakBefore = $false
            }
            "标题 3" {
                Set-ParagraphFormat $para "宋体" "Times New Roman" 12 $true $wdAlignParagraphLeft 7.8 7.8 0
                $para.PageBreakBefore = $false
            }
        }
    }
}

function Configure-BodyStyles([object]$doc) {
    $body = $doc.Styles.Item("正文")
    $body.Font.NameFarEast = "宋体"
    $body.Font.NameAscii = "Times New Roman"
    $body.Font.NameOther = "Times New Roman"
    $body.Font.Name = "Times New Roman"
    $body.Font.Size = 12
    $body.Font.Bold = $false
    $body.Font.Color = -16777216
    $body.Font.Underline = 0
    $body.ParagraphFormat.Alignment = $wdAlignParagraphJustify
    $body.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $body.ParagraphFormat.SpaceBefore = 0
    $body.ParagraphFormat.SpaceAfter = 0
    $body.ParagraphFormat.LeftIndent = 0
    $body.ParagraphFormat.FirstLineIndent = 24
    $body.ParagraphFormat.CharacterUnitLeftIndent = 0
    $body.ParagraphFormat.CharacterUnitFirstLineIndent = 0

    $heading1 = $doc.Styles.Item("标题 1")
    $heading1.Font.NameFarEast = "黑体"
    $heading1.Font.NameAscii = "Times New Roman"
    $heading1.Font.NameOther = "Times New Roman"
    $heading1.Font.Name = "Times New Roman"
    $heading1.Font.Size = 18
    $heading1.Font.Bold = $true
    $heading1.Font.Color = -16777216
    $heading1.Font.Underline = 0
    $heading1.ParagraphFormat.Alignment = $wdAlignParagraphCenter
    $heading1.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $heading1.ParagraphFormat.SpaceBefore = 31.2
    $heading1.ParagraphFormat.SpaceAfter = 15.6
    $heading1.ParagraphFormat.LeftIndent = 0
    $heading1.ParagraphFormat.FirstLineIndent = 0
    $heading1.ParagraphFormat.CharacterUnitLeftIndent = 0
    $heading1.ParagraphFormat.CharacterUnitFirstLineIndent = 0

    $heading2 = $doc.Styles.Item("标题 2")
    $heading2.Font.NameFarEast = "宋体"
    $heading2.Font.NameAscii = "Times New Roman"
    $heading2.Font.NameOther = "Times New Roman"
    $heading2.Font.Name = "Times New Roman"
    $heading2.Font.Size = 14
    $heading2.Font.Bold = $true
    $heading2.Font.Color = -16777216
    $heading2.Font.Underline = 0
    $heading2.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $heading2.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $heading2.ParagraphFormat.SpaceBefore = 7.8
    $heading2.ParagraphFormat.SpaceAfter = 7.8
    $heading2.ParagraphFormat.LeftIndent = 0
    $heading2.ParagraphFormat.FirstLineIndent = 0
    $heading2.ParagraphFormat.CharacterUnitLeftIndent = 0
    $heading2.ParagraphFormat.CharacterUnitFirstLineIndent = 0

    $heading3 = $doc.Styles.Item("标题 3")
    $heading3.Font.NameFarEast = "宋体"
    $heading3.Font.NameAscii = "Times New Roman"
    $heading3.Font.NameOther = "Times New Roman"
    $heading3.Font.Name = "Times New Roman"
    $heading3.Font.Size = 12
    $heading3.Font.Bold = $true
    $heading3.Font.Color = -16777216
    $heading3.Font.Underline = 0
    $heading3.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $heading3.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $heading3.ParagraphFormat.SpaceBefore = 7.8
    $heading3.ParagraphFormat.SpaceAfter = 7.8
    $heading3.ParagraphFormat.LeftIndent = 0
    $heading3.ParagraphFormat.FirstLineIndent = 0
    $heading3.ParagraphFormat.CharacterUnitLeftIndent = 0
    $heading3.ParagraphFormat.CharacterUnitFirstLineIndent = 0
}

function Remove-BlankParagraphsBeforeBodyChapters([object]$doc) {
    $sectionStart = $doc.Sections.Item(6).Range.Start
    for ($i = $doc.Paragraphs.Count; $i -ge 2; $i--) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -lt $sectionStart) {
            continue
        }
        $styleName = ""
        try { $styleName = $para.Range.Style.NameLocal } catch {}
        if ($styleName -ne "标题 1") {
            continue
        }
        while ($true) {
            $prev = $doc.Paragraphs.Item($i - 1)
            if ($prev.Range.Start -lt $sectionStart) {
                break
            }
            if ($prev.Range.Information($wdWithInTable)) {
                break
            }
            if (Get-ParaText $prev) {
                break
            }
            [void]$prev.Range.Delete()
            $i--
            if ($i -le 1) {
                break
            }
        }
    }
}

function Build-CaptionMetadata([object]$doc) {
    $items = New-Object System.Collections.Generic.List[object]
    $currentChapter = 0
    $figureCounter = 0
    $tableCounter = 0
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $text = Get-ParaText $para
        if (-not $text) {
            continue
        }
        $styleName = ""
        try {
            $styleName = $para.Range.Style.NameLocal
        } catch {
            $styleName = ""
        }
        if ($styleName -eq "标题 1" -and $text -match '^(?<chapter>\d+)\s') {
            $currentChapter = [int]$Matches.chapter
            $figureCounter = 0
            $tableCounter = 0
            continue
        }
        if ($text -match '^(?<kind>图|表)\s*\d+\s*[-－–—.．、]?\s*\d+\s*(?<title>.+)$') {
            if ($currentChapter -le 0) {
                continue
            }
            $kind = $Matches.kind
            $title = $Matches.title.Trim()
            if ($kind -eq "图") {
                $figureCounter++
                $number = "${currentChapter}.$figureCounter"
                $bookmark = "Fig_${currentChapter}_$figureCounter"
            } else {
                $tableCounter++
                $number = "${currentChapter}.$tableCounter"
                $bookmark = "Tab_${currentChapter}_$tableCounter"
            }
            $items.Add([pscustomobject]@{
                ParagraphIndex = $i
                Kind = $kind
                Number = $number
                Title = $title
                Bookmark = $bookmark
            })
        }
    }
    return $items
}

function Configure-TocStyles([object]$doc) {
    $usableWidth = $doc.PageSetup.PageWidth - $doc.PageSetup.LeftMargin - $doc.PageSetup.RightMargin
    $style1 = $doc.Styles.Item("TOC 1")
    $style1.Font.NameFarEast = "黑体"
    $style1.Font.NameAscii = "Times New Roman"
    $style1.Font.NameOther = "Times New Roman"
    $style1.Font.Name = "Times New Roman"
    $style1.Font.Size = 14
    $style1.Font.Bold = $false
    $style1.Font.Color = -16777216
    $style1.Font.Underline = 0
    $style1.Shading.BackgroundPatternColor = 16777215
    $style1.Shading.ForegroundPatternColor = 16777215
    $style1.Shading.Texture = 0
    $style1.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $style1.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $style1.ParagraphFormat.SpaceBefore = 0
    $style1.ParagraphFormat.SpaceAfter = 0
    $style1.ParagraphFormat.LeftIndent = 0
    $style1.ParagraphFormat.FirstLineIndent = 0
    $style1.ParagraphFormat.TabStops.ClearAll()
    [void]$style1.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)

    $style2 = $doc.Styles.Item("TOC 2")
    $style2.Font.NameFarEast = "宋体"
    $style2.Font.NameAscii = "Times New Roman"
    $style2.Font.NameOther = "Times New Roman"
    $style2.Font.Name = "Times New Roman"
    $style2.Font.Size = 12
    $style2.Font.Bold = $false
    $style2.Font.Color = -16777216
    $style2.Font.Underline = 0
    $style2.Shading.BackgroundPatternColor = 16777215
    $style2.Shading.ForegroundPatternColor = 16777215
    $style2.Shading.Texture = 0
    $style2.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $style2.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $style2.ParagraphFormat.SpaceBefore = 0
    $style2.ParagraphFormat.SpaceAfter = 0
    $style2.ParagraphFormat.LeftIndent = 21
    $style2.ParagraphFormat.FirstLineIndent = 0
    $style2.ParagraphFormat.TabStops.ClearAll()
    [void]$style2.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)

    $style3 = $doc.Styles.Item("TOC 3")
    $style3.Font.NameFarEast = "宋体"
    $style3.Font.NameAscii = "Times New Roman"
    $style3.Font.NameOther = "Times New Roman"
    $style3.Font.Name = "Times New Roman"
    $style3.Font.Size = 12
    $style3.Font.Bold = $false
    $style3.Font.Color = -16777216
    $style3.Font.Underline = 0
    $style3.Shading.BackgroundPatternColor = 16777215
    $style3.Shading.ForegroundPatternColor = 16777215
    $style3.Shading.Texture = 0
    $style3.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $style3.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $style3.ParagraphFormat.SpaceBefore = 0
    $style3.ParagraphFormat.SpaceAfter = 0
    $style3.ParagraphFormat.LeftIndent = 42
    $style3.ParagraphFormat.FirstLineIndent = 0
    $style3.ParagraphFormat.TabStops.ClearAll()
    [void]$style3.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)
}

function Apply-CaptionFormattingAndBookmarks([object]$doc, [object[]]$captions) {
    foreach ($caption in $captions) {
        $para = $doc.Paragraphs.Item($caption.ParagraphIndex)
        $content = $para.Range.Duplicate
        $content.End = [Math]::Max($content.Start, $content.End - 1)
        $content.Text = "$($caption.Kind) $($caption.Number) $($caption.Title)"
        Set-RangeFont $content "宋体" "Times New Roman" 10.5 $false
        $para.Alignment = $wdAlignParagraphCenter
        $para.LineSpacingRule = $wdLineSpace1pt5
        $para.SpaceBefore = 0
        $para.SpaceAfter = 0
        $para.FirstLineIndent = 0
        $para.LeftIndent = 0

        $bookmarkStart = $content.Start + 2
        $bookmarkEnd = $bookmarkStart + $caption.Number.Length
        Remove-ExistingBookmarkIfPresent $doc $caption.Bookmark
        [void]$doc.Bookmarks.Add($caption.Bookmark, $doc.Range($bookmarkStart, $bookmarkEnd))
    }
}

function Is-StructuralOrEmptyParagraph([string]$text) {
    if (-not $text) { return $true }
    if ($text -eq "/") { return $true }
    if ($text -match '^(图|表)\s+\d+\.\d+') { return $true }
    return $false
}

function Build-CrossRefSentenceMap([object]$doc, [object[]]$captions) {
    $map = @{}
    foreach ($caption in $captions) {
        $targetParagraphIndex = 0
        for ($i = $caption.ParagraphIndex - 1; $i -ge 1; $i--) {
            $candidate = $doc.Paragraphs.Item($i)
            $text = Get-ParaText $candidate
            if ($candidate.Range.Information($wdWithInTable)) {
                continue
            }
            if (-not (Is-StructuralOrEmptyParagraph $text)) {
                $targetParagraphIndex = $i
                break
            }
        }
        if ($targetParagraphIndex -gt 0) {
            if (-not $map.ContainsKey($targetParagraphIndex)) {
                $map[$targetParagraphIndex] = New-Object System.Collections.Generic.List[object]
            }
            $map[$targetParagraphIndex].Add([pscustomobject]@{
                Kind = $caption.Kind
                Bookmark = $caption.Bookmark
            })
        }
    }
    return $map
}

function Append-CrossRefSentences([object]$doc, [hashtable]$sentenceMap) {
    $indices = $sentenceMap.Keys | Sort-Object -Descending
    foreach ($index in $indices) {
        $para = $doc.Paragraphs.Item([int]$index)
        foreach ($item in $sentenceMap[$index]) {
            $anchor = $para.Range.Duplicate
            $anchor.End = [Math]::Max($anchor.Start, $anchor.End - 1)
            $anchor.Collapse($wdCollapseEnd)
            $token = "__XREF_$([guid]::NewGuid().ToString('N'))__"
            $prefix = "如$($item.Kind) "
            $suffix = " 所示。"
            $baseStart = $anchor.Start
            $anchor.InsertAfter($prefix + $token + $suffix)

            $tokenStart = $baseStart + $prefix.Length
            $tokenEnd = $tokenStart + $token.Length
            $fieldRange = $doc.Range($tokenStart, $tokenEnd)
            $fieldRange.Text = ""
            $field = $doc.Fields.Add($fieldRange, -1, "REF $($item.Bookmark) \h", $true)
            [void]$field.Update()
            $field.Result.Font.NameFarEast = "宋体"
            $field.Result.Font.NameAscii = "Times New Roman"
            $field.Result.Font.NameOther = "Times New Roman"
            $field.Result.Font.Size = 12
        }
    }
}

function Create-ReferenceBookmarks([object]$doc, [int]$referenceHeadingIndex) {
    $map = @{}
    for ($i = $referenceHeadingIndex + 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $text = Get-ParaText $para
        if (-not $text) {
            continue
        }
        if ($text -match '^\[(?<id>\d+)\]') {
            $id = [int]$Matches.id
            $bookmarkName = "Ref_$id"
            $matchLength = $Matches[0].Length
            $range = $doc.Range($para.Range.Start, $para.Range.Start + $matchLength)
            Remove-ExistingBookmarkIfPresent $doc $bookmarkName
            [void]$doc.Bookmarks.Add($bookmarkName, $range)
            $map[$id] = $bookmarkName

            Set-ParagraphFormat $para "宋体" "Times New Roman" 12 $false $wdAlignParagraphJustify 0 0 0
            [void]($para.PageBreakBefore = $false)
        } else {
            break
        }
    }
    return $map
}

function Replace-CitationsWithCrossRefs(
    [object]$doc,
    [int]$stopParagraphIndex,
    [hashtable]$referenceBookmarks
) {
    $regex = [regex]'\[\d+\]'
    for ($i = 1; $i -lt $stopParagraphIndex; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $styleName = ""
        try {
            $styleName = $para.Range.Style.NameLocal
        } catch {
            $styleName = ""
        }
        if ($styleName -like "TOC*") {
            continue
        }
        $text = $para.Range.Text
        $matches = $regex.Matches($text)
        for ($m = $matches.Count - 1; $m -ge 0; $m--) {
            $value = $matches[$m].Value
            $id = [int]($value -replace "[^\d]", "")
            if (-not $referenceBookmarks.ContainsKey($id)) {
                continue
            }
            $start = $para.Range.Start + $matches[$m].Index
            $end = $start + $matches[$m].Length
            $range = $doc.Range($start, $end)
            $range.Text = ""
            $field = $doc.Fields.Add($range, -1, "REF $($referenceBookmarks[$id]) \h", $true)
            [void]$field.Update()
            $field.Result.Font.Superscript = $true
            $field.Result.Font.NameFarEast = "宋体"
            $field.Result.Font.NameAscii = "Times New Roman"
            $field.Result.Font.NameOther = "Times New Roman"
            $field.Result.Font.Size = 12
        }
    }
}

function Get-TableCaptionNumber([object]$doc, [object]$table) {
    $captionParagraphIndex = 0
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -ge $table.Range.Start) {
            break
        }
        $text = Get-ParaText $para
        if ($text -match '^表\s+(?<id>\d+\.\d+)\s+') {
            $captionParagraphIndex = $i
        }
    }
    if ($captionParagraphIndex -gt 0) {
        $text = Get-ParaText $doc.Paragraphs.Item($captionParagraphIndex)
        if ($text -match '^表\s+(?<id>\d+\.\d+)\s+') {
            return $Matches.id
        }
    }
    return ""
}

function Try-SplitCrossPageTables([object]$word, [object]$doc) {
    $changed = $false
    for ($t = $doc.Tables.Count; $t -ge 1; $t--) {
        $table = $doc.Tables.Item($t)
        $startPage = $table.Range.Information($wdInfoPage)
        $endPage = $table.Range.Characters.Last.Information($wdInfoPage)
        if ($startPage -eq $endPage) {
            continue
        }
        $captionNumber = Get-TableCaptionNumber $doc $table
        if (-not $captionNumber) {
            continue
        }

        $splitAtRow = 0
        $previousPage = $table.Rows.Item(1).Range.Information($wdInfoPage)
        for ($rowIndex = 2; $rowIndex -le $table.Rows.Count; $rowIndex++) {
            $rowPage = $table.Rows.Item($rowIndex).Range.Information($wdInfoPage)
            if ($rowPage -gt $previousPage) {
                $splitAtRow = $rowIndex
                break
            }
            $previousPage = $rowPage
        }
        if ($splitAtRow -le 1) {
            continue
        }

        $headerTexts = @()
        for ($cellIndex = 1; $cellIndex -le $table.Rows.Item(1).Cells.Count; $cellIndex++) {
            $cellText = Get-RangeText $table.Rows.Item(1).Cells.Item($cellIndex).Range
            $cellText = $cellText.TrimEnd([char]13, [char]7)
            $headerTexts += $cellText
        }

        $table.Rows.Item($splitAtRow).Range.Select()
        $word.Selection.SplitTable()

        $newTable = $doc.Tables.Item($t + 1)
        [void]$newTable.Rows.Add($newTable.Rows.Item(1))
        for ($cellIndex = 1; $cellIndex -le $newTable.Rows.Item(1).Cells.Count; $cellIndex++) {
            if ($cellIndex -le $headerTexts.Count) {
                $newCellRange = $newTable.Rows.Item(1).Cells.Item($cellIndex).Range
                $newCellRange.Text = $headerTexts[$cellIndex - 1]
            }
        }

        $newTable.Rows.Item(1).HeadingFormat = $true
        $captionRange = $doc.Range($newTable.Range.Start, $newTable.Range.Start)
        $captionRange.InsertParagraphBefore()
        $captionPara = $doc.Range($newTable.Range.Start - 1, $newTable.Range.Start - 1).Paragraphs.Item(1)
        $captionPara.Range.Text = "续表 $captionNumber"
        Set-ParagraphFormat $captionPara "宋体" "Times New Roman" 10.5 $false $wdAlignParagraphCenter 0 0 0

        $changed = $true
    }
    return $changed
}

function Format-TocParagraphs([object]$doc, [object]$toc) {
    $usableWidth = $doc.PageSetup.PageWidth - $doc.PageSetup.LeftMargin - $doc.PageSetup.RightMargin
    foreach ($para in $doc.Paragraphs) {
        if ($para.Range.Start -lt $toc.Range.Start -or $para.Range.Start -gt $toc.Range.End) {
            continue
        }
        $text = Get-ParaText $para
        if (-not $text) {
            continue
        }

        $para.LineSpacingRule = $wdLineSpace1pt5
        $para.SpaceBefore = 0
        $para.SpaceAfter = 0
        $para.Range.ParagraphFormat.TabStops.ClearAll()
        [void]$para.Range.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)
        Clear-ParagraphShading $para

        if ($text -match '^\d+\.\d+\.\d+\s') {
            $para.Range.Style = $doc.Styles.Item("TOC 3")
            Set-RangeFont $para.Range "宋体" "Times New Roman" 12 $false
            $para.Alignment = $wdAlignParagraphLeft
            $para.LeftIndent = 42
            $para.FirstLineIndent = 0
        } elseif ($text -match '^\d+\.\d+\s') {
            $para.Range.Style = $doc.Styles.Item("TOC 2")
            Set-RangeFont $para.Range "宋体" "Times New Roman" 12 $false
            $para.Alignment = $wdAlignParagraphLeft
            $para.LeftIndent = 21
            $para.FirstLineIndent = 0
        } elseif ($text -match '^\d+\s' -or $text -eq '参考文献') {
            $para.Range.Style = $doc.Styles.Item("TOC 1")
            Set-RangeFont $para.Range "黑体" "Times New Roman" 14 $false
            $para.Alignment = $wdAlignParagraphLeft
            $para.LeftIndent = 0
            $para.FirstLineIndent = 0
        }
    }
}

function Build-Toc([object]$doc) {
    $sectionRange = Get-TocEditableRange $doc
    $sectionRange.Text = "目    录`r`r"

    $sectionParagraphs = Get-SectionParagraphs $doc 5
    $titlePara = $sectionParagraphs[0]
    Set-ParagraphFormat $titlePara "黑体" "Times New Roman" 22 $false $wdAlignParagraphCenter 15.6 7.8 0

    $insertRange = $sectionParagraphs[1].Range.Duplicate
    $insertRange.End = $insertRange.Start
    $toc = $doc.TablesOfContents.Add($insertRange, $true, 1, 3, $false, "", $true, $true, "", $false, $true, $true)
    $toc.TabLeader = $wdTabLeaderDots
    [void]$toc.Update()
    Format-TocParagraphs $doc $toc
}

function Normalize-TocParagraphStyles([object]$doc) {
    if ($doc.TablesOfContents.Count -lt 1) {
        return
    }
    $toc = $doc.TablesOfContents.Item(1)
    Format-TocParagraphs $doc $toc

    $usableWidth = $doc.PageSetup.PageWidth - $doc.PageSetup.LeftMargin - $doc.PageSetup.RightMargin
    foreach ($para in (Get-SectionParagraphs $doc 5)) {
        Clear-ParagraphShading $para
        $text = Get-ParaText $para
        if (-not $text -or $text -eq "目    录") {
            continue
        }
        if ($text -match '^\d+\.\d+\.\d+\s') {
            Clear-ParagraphShading $para
            $para.Range.ParagraphFormat.TabStops.ClearAll()
            [void]$para.Range.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)
            try { $para.Range.Style = $doc.Styles.Item("TOC 3") } catch {}
            Set-RangeFont $para.Range "宋体" "Times New Roman" 12 $false
            $para.Alignment = $wdAlignParagraphLeft
            $para.LeftIndent = 42
            $para.FirstLineIndent = 0
        } elseif ($text -match '^\d+\.\d+\s') {
            Clear-ParagraphShading $para
            $para.Range.ParagraphFormat.TabStops.ClearAll()
            [void]$para.Range.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)
            try { $para.Range.Style = $doc.Styles.Item("TOC 2") } catch {}
            Set-RangeFont $para.Range "宋体" "Times New Roman" 12 $false
            $para.Alignment = $wdAlignParagraphLeft
            $para.LeftIndent = 21
            $para.FirstLineIndent = 0
        } elseif ($text -match '^\d+\s' -or $text -eq '参考文献') {
            Clear-ParagraphShading $para
            $para.Range.ParagraphFormat.TabStops.ClearAll()
            [void]$para.Range.ParagraphFormat.TabStops.Add($usableWidth, $wdAlignTabRight, $wdTabLeaderDots)
            Set-RangeFont $para.Range "黑体" "Times New Roman" 14 $false
            $para.Alignment = $wdAlignParagraphLeft
            $para.LeftIndent = 0
            $para.FirstLineIndent = 0
            try { $para.Range.Style = $doc.Styles.Item("TOC 1") } catch {}
        }
        $para.LineSpacingRule = $wdLineSpace1pt5
        $para.SpaceBefore = 0
        $para.SpaceAfter = 0
    }
}

function Trim-TocTrailingBlankParagraphs([object]$doc) {
    $sectionStart = $doc.Sections.Item(5).Range.Start
    $sectionEnd = $doc.Sections.Item(5).Range.End
    for ($i = $doc.Paragraphs.Count; $i -ge 1; $i--) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -ge $sectionEnd) {
            continue
        }
        if ($para.Range.Start -lt $sectionStart) {
            break
        }
        Clear-ParagraphShading $para
        $text = Get-ParaText $para
        if ($text -and $text -ne "目    录") {
            break
        }
        if ($para.Range.End -ge $sectionEnd) {
            continue
        }
        if (-not $text) {
            try {
                [void]$para.Range.Delete()
            } catch {
            }
        }
    }
}

function Update-FieldsInSection([object]$doc, [int]$sectionIndex) {
    $range = Get-EditableSectionRange $doc $sectionIndex
    foreach ($field in $range.Fields) {
        try { [void]$field.Update() } catch {}
    }
}

function Update-AllFields([object]$doc) {
    foreach ($field in $doc.Fields) {
        try { [void]$field.Update() } catch {}
    }
}

if (Test-Path -LiteralPath $OutputDocx) {
    Remove-Item -LiteralPath $OutputDocx -Force
}
if (Test-Path -LiteralPath $OutputDoc) {
    Remove-Item -LiteralPath $OutputDoc -Force
}

$word = $null
$sourceDoc = $null
$doc = $null
$docForLegacy = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $sourceDoc = $word.Documents.Open($SourcePath, $false, $true)
    $doc = $word.Documents.Open($TemplatePath, $false, $false)
    try { $word.ActiveWindow.View.FieldShading = 0 } catch {}
    $doc.SaveAs([ref]$OutputDocx, [ref]$wdFormatDocumentDefault)

    $coverInfo = Get-CoverMetadata $sourceDoc
    Fill-Cover $doc $coverInfo
    Tighten-CoverLayout $doc $coverInfo.MonthText
    Remove-CoverMarker $doc

    $cnAbsTitleIndex = Find-ParagraphIndexByExactText $sourceDoc $coverInfo.CnTitle ($coverInfo.DeclarationIndex + 1)
    $enAbsTitleIndex = Find-ParagraphIndexByExactText $sourceDoc $coverInfo.EnTitle ($cnAbsTitleIndex + 1)
    $tocTitleIndex = Find-ParagraphIndexByExactText $sourceDoc "目    录" ($enAbsTitleIndex + 1)
    $bodyStartIndex = Find-ParagraphIndexByStyleAndRegex $sourceDoc "标题 1" '^\d+\s+' ($tocTitleIndex + 1)

    $declarationRange = Get-TrimmedRangeBetweenParagraphs $sourceDoc $coverInfo.DeclarationIndex $cnAbsTitleIndex
    $cnAbstractRange = Get-TrimmedRangeBetweenParagraphs $sourceDoc $cnAbsTitleIndex $enAbsTitleIndex
    $enAbstractRange = Get-TrimmedRangeBetweenParagraphs $sourceDoc $enAbsTitleIndex $tocTitleIndex
    $bodyRange = Get-RangeBetweenParagraphs $sourceDoc $bodyStartIndex ($sourceDoc.Paragraphs.Count + 1)

    Replace-SectionContent $doc 2 $declarationRange
    Replace-SectionContent $doc 3 $cnAbstractRange
    Replace-SectionContent $doc 4 $enAbstractRange
    Replace-SectionContent $doc 6 $bodyRange

    Configure-BodyStyles $doc
    Apply-FrontMatterFormatting $doc $coverInfo.CnTitle $coverInfo.EnTitle
    Normalize-FrontMatter $doc
    Normalize-BodyHeadings $doc
    Remove-BlankParagraphsBeforeBodyChapters $doc

    $captions = Build-CaptionMetadata $doc
    Apply-CaptionFormattingAndBookmarks $doc $captions
    $sentenceMap = Build-CrossRefSentenceMap $doc $captions
    Append-CrossRefSentences $doc $sentenceMap

    $referenceHeadingIndex = Find-ParagraphIndexByExactText $doc "参考文献"
    $referenceBookmarks = Create-ReferenceBookmarks $doc $referenceHeadingIndex
    Replace-CitationsWithCrossRefs $doc $referenceHeadingIndex $referenceBookmarks

    Build-Toc $doc
    $doc.Repaginate()
    $tableSplit = Try-SplitCrossPageTables $word $doc
    if ($tableSplit) {
        $doc.Repaginate()
    }
    Update-AllFields $doc
    Configure-TocStyles $doc
    foreach ($toc in $doc.TablesOfContents) {
        try { [void]$toc.Update() } catch {}
    }
    Update-AllFields $doc
    Configure-TocStyles $doc
    Normalize-TocParagraphStyles $doc
    Ensure-TocBodySectionBreak $doc
    Configure-Headers $doc
    Configure-PageNumbers $doc
    Normalize-BodyHeadings $doc
    Remove-BlankParagraphsBeforeBodyChapters $doc
    $doc.Repaginate()
    Trim-TocTrailingBlankParagraphs $doc
    $doc.Save()
    $doc.Close($false)
    $doc = $null

    $docForLegacy = $word.Documents.Open($OutputDocx, $false, $false)
    try { $word.ActiveWindow.View.FieldShading = 0 } catch {}
    $docForLegacy.SaveAs([ref]$OutputDoc, [ref]$wdFormatDocument97)
    $docForLegacy.Close($false)
    $docForLegacy = $null

    Write-Output "Created: $OutputDocx"
    Write-Output "Created: $OutputDoc"
} finally {
    if ($docForLegacy -ne $null) {
        try { $docForLegacy.Close($false) } catch {}
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($docForLegacy) | Out-Null
    }
    if ($doc -ne $null) {
        try { $doc.Close($false) } catch {}
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
    if ($sourceDoc -ne $null) {
        try { $sourceDoc.Close($false) } catch {}
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($sourceDoc) | Out-Null
    }
    if ($word -ne $null) {
        try { $word.Quit() } catch {}
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
