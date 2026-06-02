param(
    [string]$SourcePath = "D:\计算机本科毕业论文\论文极限版.docx",
    [string]$TemplatePath = "D:\计算机本科毕业论文\附件1：江苏大学毕业设计（论文）通用格式模板（理工农医教）-2026版.doc",
    [string]$OutputDocx = "D:\计算机本科毕业论文\论文极限格式规范版.docx",
    [string]$OutputDoc = "D:\计算机本科毕业论文\论文极限格式规范版.doc"
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
$wdStatisticPages = 2
$wdInfoPage = 3
$wdWithInTable = 12
$wdFormatDocumentDefault = 16
$wdFormatDocument97 = 0

function Get-ParaText([object]$para) {
    return (($para.Range.Text -replace "[`r`a]", "")).Trim()
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

function Find-ParagraphIndexByExactTextAndStyle(
    [object]$doc,
    [string]$text,
    [string]$styleName,
    [int]$startIndex = 1
) {
    for ($i = $startIndex; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $currentText = Get-ParaText $para
        $currentStyle = ""
        try {
            $currentStyle = $para.Range.Style.NameLocal
        } catch {
            $currentStyle = ""
        }
        if ($currentText -eq $text -and $currentStyle -eq $styleName) {
            return $i
        }
    }
    throw "Paragraph not found with style ${styleName}: $text"
}

function Find-ParagraphIndexByRegexAndStyle(
    [object]$doc,
    [string]$pattern,
    [string]$styleName,
    [int]$startIndex = 1
) {
    for ($i = $startIndex; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $currentText = Get-ParaText $para
        $currentStyle = ""
        try {
            $currentStyle = $para.Range.Style.NameLocal
        } catch {
            $currentStyle = ""
        }
        if ($currentStyle -eq $styleName -and $currentText -match $pattern) {
            return $i
        }
    }
    throw "Paragraph pattern not found with style ${styleName}: $pattern"
}

function Find-ParagraphIndexByRegexPreferStyle(
    [object]$doc,
    [string]$pattern,
    [string]$styleName,
    [int]$startIndex = 1
) {
    $fallbackIndex = 0
    for ($i = $startIndex; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $currentText = Get-ParaText $para
        if ($currentText -notmatch $pattern) {
            continue
        }
        $fallbackIndex = $i
        $currentStyle = ""
        try {
            $currentStyle = $para.Range.Style.NameLocal
        } catch {
            $currentStyle = ""
        }
        if ($currentStyle -eq $styleName) {
            return $i
        }
    }
    if ($fallbackIndex -gt 0) {
        return $fallbackIndex
    }
    throw "Paragraph pattern not found with preferred style ${styleName}: $pattern"
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
}

function Set-HeadingStyles([object]$doc) {
    foreach ($para in $doc.Paragraphs) {
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
        if ($styleName -eq "标题 1") {
            Set-ParagraphFormat $para "黑体" "Times New Roman" 18 $true $wdAlignParagraphCenter 31.2 15.6 0
            [void]($para.PageBreakBefore = $true)
        } elseif ($styleName -eq "标题 2") {
            Set-ParagraphFormat $para "宋体" "Times New Roman" 14 $true $wdAlignParagraphLeft 7.8 7.8 0
            [void]($para.PageBreakBefore = $false)
        } elseif ($styleName -eq "标题 3") {
            Set-ParagraphFormat $para "宋体" "Times New Roman" 12 $true $wdAlignParagraphLeft 7.8 7.8 0
            [void]($para.PageBreakBefore = $false)
        }
    }
}

function Replace-CoverWithTemplate(
    [object]$doc,
    [object]$templateDoc,
    [string]$thesisCnTitle,
    [string]$thesisEnTitle,
    [string]$monthText
) {
    $currentCoverRange = $doc.Range($doc.Sections.Item(1).Range.Start, $doc.Sections.Item(2).Range.Start)
    $templateCoverRange = $templateDoc.Range($templateDoc.Sections.Item(1).Range.Start, $templateDoc.Sections.Item(2).Range.Start)
    $currentCoverRange.FormattedText = $templateCoverRange.FormattedText

    if ((Get-ParaText $doc.Paragraphs.Item(1)) -eq "附件1") {
        $doc.Paragraphs.Item(1).Range.Delete()
    }

    $statementStart = $doc.Sections.Item(2).Range.Start
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -ge $statementStart) {
            break
        }
        $text = Get-ParaText $para
        switch ($text) {
            "中文论文题目（二号黑体，居中，可分两行）" {
                $para.Range.Text = $thesisCnTitle + "`r"
            }
            "英文论文题目（16pt Time New Roman，加粗）" {
                $para.Range.Text = $thesisEnTitle + "`r"
            }
        }
        if ($text -like '*小四宋体*' -and $text -like '*年*' -and $text -like '*月*') {
            $para.Range.Text = $monthText + "`r"
        }
    }

    $coverTable = $doc.Tables.Item(1)
    if ($coverTable.Range.Start -lt $statementStart) {
        for ($row = 1; $row -le $coverTable.Rows.Count; $row++) {
            $coverTable.Cell($row, 2).Range.Text = ""
        }
    }
}

function Mark-BodyStart([object]$doc) {
    $chapterOneIndex = Find-ParagraphIndexByRegexAndStyle $doc '^1\s*绪论$' "标题 1" 1
    Remove-ExistingBookmarkIfPresent $doc "BodyStartHeading"
    [void]$doc.Bookmarks.Add("BodyStartHeading", $doc.Range($doc.Paragraphs.Item($chapterOneIndex).Range.Start, $doc.Paragraphs.Item($chapterOneIndex).Range.Start))
}

function Get-BodyStartParagraphIndex([object]$doc) {
    if (-not $doc.Bookmarks.Exists("BodyStartHeading")) {
        return Find-ParagraphIndexByRegexAndStyle $doc '^1\s*绪论$' "标题 1" 1
    }
    $bookmarkStart = $doc.Bookmarks.Item("BodyStartHeading").Range.Start
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -le $bookmarkStart -and $para.Range.End -ge $bookmarkStart) {
            return $i
        }
    }
    return Find-ParagraphIndexByRegexAndStyle $doc '^1\s*绪论$' "标题 1" 1
}

function Clear-ParagraphShading([object]$para) {
    try {
        $para.Shading.BackgroundPatternColor = -16777216
        $para.Shading.ForegroundPatternColor = -16777216
        $para.Shading.Texture = 0
    } catch {
    }
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

function Finalize-FrontMatterBreaks([object]$doc) {
    if ($doc.TablesOfContents.Count -lt 1) {
        return
    }

    $tocEnd = $doc.TablesOfContents.Item(1).Range.End
    $chapterOneIndex = Get-BodyStartParagraphIndex $doc
    if ($doc.Paragraphs.Item($chapterOneIndex).Range.Start -le $tocEnd) {
        $tocHeadingIndex = Find-ParagraphIndexByRegex $doc '^目\s*录$'
        $chapterOneIndex = Find-ParagraphIndexByRegexPreferStyle $doc '^1\s*绪论$' "标题 1" ($tocHeadingIndex + 1)
    }

    [void]($doc.Paragraphs.Item($chapterOneIndex).PageBreakBefore = $false)

    $between = $doc.Range($tocEnd, $doc.Paragraphs.Item($chapterOneIndex).Range.Start)
    $between.Text = ""

    $insertRange = $doc.Range($doc.TablesOfContents.Item(1).Range.End, $doc.TablesOfContents.Item(1).Range.End)
    $insertRange.InsertBreak($wdSectionBreakNextPage)
}

function Clear-RangeBetweenParagraphs([object]$doc, [int]$afterIndex, [int]$beforeIndex) {
    $range = $doc.Range($doc.Paragraphs.Item($afterIndex).Range.End, $doc.Paragraphs.Item($beforeIndex).Range.Start)
    $range.Text = ""
}

function Insert-SectionBreakBetweenParagraphs([object]$doc, [int]$afterIndex, [int]$beforeIndex) {
    Clear-RangeBetweenParagraphs $doc $afterIndex $beforeIndex
    $insertRange = $doc.Range($doc.Paragraphs.Item($afterIndex).Range.End, $doc.Paragraphs.Item($afterIndex).Range.End)
    $insertRange.InsertBreak($wdSectionBreakNextPage)
}

function Set-HeaderText([object]$section, [string]$leftText, [string]$rightText) {
    $header = $section.Headers.Item($wdHeaderFooterPrimary)
    $header.LinkToPrevious = $false
    $header.Range.Text = ""
    if ($leftText) {
        $usableWidth = $section.PageSetup.PageWidth - $section.PageSetup.LeftMargin - $section.PageSetup.RightMargin
        $header.Range.ParagraphFormat.TabStops.ClearAll()
        [void]$header.Range.ParagraphFormat.TabStops.Add($usableWidth, 2, 0)
        $header.Range.Text = "$leftText`t$rightText"
        $header.Range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    } else {
        $header.Range.Text = $rightText
        $header.Range.ParagraphFormat.Alignment = $wdAlignParagraphRight
    }
    Set-RangeFont $header.Range "宋体" "Times New Roman" 9 $false
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

function Apply-PageSetupFromTemplate([object]$doc, [object]$templateDoc) {
    foreach ($section in $doc.Sections) {
        $section.PageSetup.TopMargin = $templateDoc.PageSetup.TopMargin
        $section.PageSetup.BottomMargin = $templateDoc.PageSetup.BottomMargin
        $section.PageSetup.LeftMargin = $templateDoc.PageSetup.LeftMargin
        $section.PageSetup.RightMargin = $templateDoc.PageSetup.RightMargin
        $section.PageSetup.HeaderDistance = $templateDoc.PageSetup.HeaderDistance
        $section.PageSetup.FooterDistance = $templateDoc.PageSetup.FooterDistance
        $section.PageSetup.PageWidth = $templateDoc.PageSetup.PageWidth
        $section.PageSetup.PageHeight = $templateDoc.PageSetup.PageHeight
    }
}

function Remove-ExistingBookmarkIfPresent([object]$doc, [string]$bookmarkName) {
    if ($doc.Bookmarks.Exists($bookmarkName)) {
        $doc.Bookmarks.Item($bookmarkName).Delete()
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

function Rebuild-TableOfContents([object]$doc) {
    $tocHeadingIndex = Find-ParagraphIndexByRegex $doc '^目\s*录$'
    $chapterOneIndex = Get-BodyStartParagraphIndex $doc
    $tocHeading = $doc.Paragraphs.Item($tocHeadingIndex)
    if ($doc.Paragraphs.Item($chapterOneIndex).Range.Start -le $tocHeading.Range.End) {
        $chapterOneIndex = Find-ParagraphIndexByRegexPreferStyle $doc '^1\s*绪论$' "标题 1" ($tocHeadingIndex + 1)
    }

    $tocHeading.Range.Style = $doc.Styles.Item("正文")
    Set-ParagraphFormat $tocHeading "黑体" "Times New Roman" 22 $false $wdAlignParagraphCenter 15.6 7.8 0

    if ($doc.TablesOfContents.Count -gt 0) {
        $doc.TablesOfContents.Item(1).Delete()
        $chapterOneIndex = Get-BodyStartParagraphIndex $doc
    }
    for ($i = $chapterOneIndex - 1; $i -gt $tocHeadingIndex; $i--) {
        $doc.Paragraphs.Item($i).Range.Delete()
    }

    $insertRange = $doc.Range($tocHeading.Range.End, $tocHeading.Range.End)
    $toc = $doc.TablesOfContents.Add($insertRange, $true, 1, 3, $false, "", $true, $true, "", $false, $true, $true)
    $toc.TabLeader = $wdTabLeaderDots
    [void]$toc.Update()
    Format-TocParagraphs $doc $toc
}

function Normalize-TocParagraphStyles([object]$doc) {
    if ($doc.TablesOfContents.Count -lt 1) {
        return
    }
    Format-TocParagraphs $doc $doc.TablesOfContents.Item(1)
}

function Apply-FrontMatterFormatting([object]$doc, [string]$thesisCnTitle, [string]$thesisEnTitle, [string]$monthText) {
    $coverCnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisCnTitle
    $coverEnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisEnTitle ($coverCnTitleIndex + 1)
    try {
        $monthIndex = Find-ParagraphIndexByRegex $doc '^\d{4}年\d{1,2}月$' ($coverEnTitleIndex + 1)
    } catch {
        $monthIndex = Find-ParagraphIndexByRegex $doc '^年\s*月' ($coverEnTitleIndex + 1)
    }
    $statementTitleIndex = Find-ParagraphIndexByExactText $doc "毕业设计（论文）原创性声明" ($monthIndex + 1)
    $abstractCnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisCnTitle ($statementTitleIndex + 1)
    $abstractEnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisEnTitle ($abstractCnTitleIndex + 1)
    $tocHeadingIndex = Find-ParagraphIndexByRegex $doc '^目\s*录$' ($abstractEnTitleIndex + 1)

    $coverCnPara = $doc.Paragraphs.Item($coverCnTitleIndex)
    $coverCnPara.Range.Text = $thesisCnTitle + "`r"
    Set-RangeFont $coverCnPara.Range "黑体" "黑体" 22 $false
    $coverCnPara.Alignment = $wdAlignParagraphCenter
    $coverCnPara.SpaceBefore = 0
    $coverCnPara.SpaceAfter = 0

    $coverEnPara = $doc.Paragraphs.Item($coverEnTitleIndex)
    $coverEnPara.Range.Text = $thesisEnTitle + "`r"
    Set-RangeFont $coverEnPara.Range "宋体" "Times New Roman" 16 $true
    $coverEnPara.Alignment = $wdAlignParagraphCenter
    $coverEnPara.SpaceBefore = 0
    $coverEnPara.SpaceAfter = 0

    $monthPara = $doc.Paragraphs.Item($monthIndex)
    $monthPara.Range.Text = $monthText + "`r"
    Set-RangeFont $monthPara.Range "宋体" "Times New Roman" 12 $false
    $monthPara.Alignment = $wdAlignParagraphCenter
    $monthPara.SpaceBefore = 0
    $monthPara.SpaceAfter = 0

    $statementPara = $doc.Paragraphs.Item($statementTitleIndex)
    Set-RangeFont $statementPara.Range "黑体" "黑体" 16 $false
    $statementPara.Alignment = $wdAlignParagraphCenter
    $statementPara.SpaceBefore = 0
    $statementPara.SpaceAfter = 0

    Set-ParagraphFormat $doc.Paragraphs.Item($abstractCnTitleIndex) "黑体" "黑体" 16 $false $wdAlignParagraphCenter 15.6 7.8 0
    Set-ParagraphFormat $doc.Paragraphs.Item($abstractCnTitleIndex + 1) "宋体" "Times New Roman" 12 $false $wdAlignParagraphCenter 0 0 0
    Set-ParagraphFormat $doc.Paragraphs.Item($abstractCnTitleIndex + 2) "宋体" "Times New Roman" 12 $false $wdAlignParagraphCenter 0 0 0
    Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractCnTitleIndex + 3) 3 "宋体" "Times New Roman" 12 $false "黑体" "黑体" 14 $false $wdAlignParagraphJustify 15.6 0 28
    Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractCnTitleIndex + 4) 4 "宋体" "Times New Roman" 12 $false "黑体" "黑体" 14 $false $wdAlignParagraphLeft 15.6 0 0

    Set-ParagraphFormat $doc.Paragraphs.Item($abstractEnTitleIndex) "黑体" "Times New Roman" 14 $true $wdAlignParagraphCenter 15.6 15.6 0
    Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractEnTitleIndex + 1) 8 "宋体" "Times New Roman" 12 $false "宋体" "Times New Roman" 12 $true $wdAlignParagraphJustify 0 0 12
    Format-ParagraphWithPrefix $doc.Paragraphs.Item($abstractEnTitleIndex + 2) 10 "宋体" "Times New Roman" 12 $false "宋体" "Times New Roman" 12 $true $wdAlignParagraphLeft 15.6 0 0

    Set-ParagraphFormat $doc.Paragraphs.Item($tocHeadingIndex) "黑体" "Times New Roman" 22 $false $wdAlignParagraphCenter 15.6 7.8 0
}

function Build-RequiredSections([object]$doc, [string]$thesisCnTitle, [string]$thesisEnTitle) {
    try {
        $monthIndex = Find-ParagraphIndexByRegex $doc '^\d{4}年\d{1,2}月$'
    } catch {
        $monthIndex = Find-ParagraphIndexByRegex $doc '^年\s*月'
    }
    $statementTitleIndex = Find-ParagraphIndexByExactText $doc "毕业设计（论文）原创性声明" ($monthIndex + 1)

    $dateIndex = Find-ParagraphIndexByRegex $doc '^日期：\s*年\s*月\s*日$' $statementTitleIndex
    $abstractCnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisCnTitle ($dateIndex + 1)
    Insert-SectionBreakBetweenParagraphs $doc $dateIndex $abstractCnTitleIndex

    $cnKeywordIndex = Find-ParagraphIndexByRegex $doc '^关键词：' $abstractCnTitleIndex
    $abstractEnTitleIndex = Find-ParagraphIndexByExactText $doc $thesisEnTitle ($cnKeywordIndex + 1)
    Insert-SectionBreakBetweenParagraphs $doc $cnKeywordIndex $abstractEnTitleIndex

    $enKeywordIndex = Find-ParagraphIndexByRegex $doc '^KEY WORDS:' $abstractEnTitleIndex
    $tocHeadingIndex = Find-ParagraphIndexByRegex $doc '^目\s*录$' ($enKeywordIndex + 1)
    Insert-SectionBreakBetweenParagraphs $doc $enKeywordIndex $tocHeadingIndex

    $chapterOneIndex = Get-BodyStartParagraphIndex $doc
    [void]($doc.Paragraphs.Item($chapterOneIndex).PageBreakBefore = $false)
}

function Configure-HeadersAndFooters([object]$doc) {
    if ($doc.Sections.Count -lt 6) {
        throw "Expected at least 6 sections after rebuilding front matter, got $($doc.Sections.Count)"
    }

    $doc.Sections.Item(1).Headers.Item($wdHeaderFooterPrimary).Range.Text = ""
    $doc.Sections.Item(1).Footers.Item($wdHeaderFooterPrimary).Range.Text = ""

    Set-HeaderText $doc.Sections.Item(2) "原创性声明" "江苏大学本科毕业设计（论文）"
    Set-FooterPageNumbers $doc.Sections.Item(2) $wdPageNumberStyleUppercaseRoman $true 1

    Set-HeaderText $doc.Sections.Item(3) "摘要" "江苏大学本科毕业设计（论文）"
    Set-FooterPageNumbers $doc.Sections.Item(3) $wdPageNumberStyleUppercaseRoman $true 2

    Set-HeaderText $doc.Sections.Item(4) "ABSTRACT" "江苏大学本科毕业设计（论文）"
    Set-FooterPageNumbers $doc.Sections.Item(4) $wdPageNumberStyleUppercaseRoman $true 3

    Set-HeaderText $doc.Sections.Item(5) "目录" "江苏大学本科毕业设计（论文）"
    Set-FooterPageNumbers $doc.Sections.Item(5) $wdPageNumberStyleUppercaseRoman $true 4

    Set-HeaderText $doc.Sections.Item(6) "" "江苏大学本科毕业设计（论文）"
    Set-FooterPageNumbers $doc.Sections.Item(6) $wdPageNumberStyleArabic $true 1
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

$thesisCnTitle = "基于CNN的手写数学公式识别系统的设计与实现"
$thesisEnTitle = "Design and Implementation of a CNN-Based Handwritten Mathematical Expression Recognition System"
$coverMonthText = "2026年6月"

if (-not (Test-Path -LiteralPath $SourcePath)) {
    throw "Source file not found: $SourcePath"
}
if (-not (Test-Path -LiteralPath $TemplatePath)) {
    throw "Template file not found: $TemplatePath"
}

Copy-Item -LiteralPath $SourcePath -Destination $OutputDocx -Force

$word = $null
$doc = $null
$templateDoc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($OutputDocx, $false, $false)
    $templateDoc = $word.Documents.Open($TemplatePath, $false, $true)

    Apply-PageSetupFromTemplate $doc $templateDoc
    Replace-CoverWithTemplate $doc $templateDoc $thesisCnTitle $thesisEnTitle $coverMonthText
    Set-HeadingStyles $doc
    Mark-BodyStart $doc
    Apply-FrontMatterFormatting $doc $thesisCnTitle $thesisEnTitle $coverMonthText

    $captions = Build-CaptionMetadata $doc
    Apply-CaptionFormattingAndBookmarks $doc $captions
    $sentenceMap = Build-CrossRefSentenceMap $doc $captions
    Append-CrossRefSentences $doc $sentenceMap

    $referenceHeadingIndex = Find-ParagraphIndexByExactText $doc "参考文献"
    $referenceBookmarks = Create-ReferenceBookmarks $doc $referenceHeadingIndex
    Replace-CitationsWithCrossRefs $doc $referenceHeadingIndex $referenceBookmarks

    Rebuild-TableOfContents $doc
    Build-RequiredSections $doc $thesisCnTitle $thesisEnTitle
    Finalize-FrontMatterBreaks $doc
    Apply-PageSetupFromTemplate $doc $templateDoc
    Configure-HeadersAndFooters $doc

    $doc.Repaginate()
    $tableSplit = Try-SplitCrossPageTables $word $doc
    if ($tableSplit) {
        $doc.Repaginate()
    }

    foreach ($toc in $doc.TablesOfContents) {
        [void]$toc.Update()
    }
    Normalize-TocParagraphStyles $doc
    foreach ($field in $doc.Fields) {
        try {
            $field.Update() | Out-Null
        } catch {
        }
    }
    Finalize-FrontMatterBreaks $doc
    $doc.Repaginate()

    $doc.SaveAs([ref]$OutputDocx, [ref]$wdFormatDocumentDefault)
    $doc.SaveAs([ref]$OutputDoc, [ref]$wdFormatDocument97)

    $pages = $doc.ComputeStatistics($wdStatisticPages)
    Write-Output "DONE"
    Write-Output "OUTPUT_DOCX=$OutputDocx"
    Write-Output "OUTPUT_DOC=$OutputDoc"
    Write-Output "PAGES=$pages"
    Write-Output "SECTIONS=$($doc.Sections.Count)"
    Write-Output "TABLE_SPLIT=$tableSplit"
} finally {
    if ($templateDoc -ne $null) {
        $templateDoc.Close(0)
    }
    if ($doc -ne $null) {
        $doc.Close(0)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
}
