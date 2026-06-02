(function () {
    var canvas = document.getElementById('sketch-canvas');
    if (!canvas) {
        return;
    }

    var ctx = canvas.getContext('2d');
    var drawing = false;
    var hasStroke = false;
    var penWidthInput = document.getElementById('pen-width');
    var penWidthValue = document.getElementById('pen-width-value');
    var resultBox = document.getElementById('sketch-result');
    var wordLinearBox = document.getElementById('sketch-word-linear');
    var historyHint = document.getElementById('sketch-history-hint');
    var correctionButton = document.getElementById('sketch-correction-button');
    var correctionModal = document.getElementById('sketch-correction-modal');
    var correctionOptions = document.getElementById('sketch-correction-options');
    var correctionInput = document.getElementById('sketch-correction-input');
    var currentFormulaPngUrl = '';
    var currentWordMathml = '';
    var currentHistoryId = null;
    var currentCorrectionSuggestions = [];

    function syncPenWidth() {
        var width = Number(penWidthInput ? penWidthInput.value : 12);
        ctx.lineWidth = width;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000000';
        if (penWidthValue) {
            penWidthValue.textContent = String(width);
        }
    }

    function fillBackground() {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
    }

    function initializeCanvas() {
        fillBackground();
        syncPenWidth();
    }

    function getPos(event) {
        var rect = canvas.getBoundingClientRect();
        var scaleX = canvas.width / rect.width;
        var scaleY = canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    function startDraw(event) {
        drawing = true;
        hasStroke = true;
        var pos = getPos(event);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        ctx.lineTo(pos.x + 0.01, pos.y + 0.01);
        ctx.stroke();
    }

    function moveDraw(event) {
        if (!drawing) {
            return;
        }
        var pos = getPos(event);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function endDraw() {
        drawing = false;
    }

    canvas.addEventListener('pointerdown', function (event) {
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        startDraw(event);
    });

    canvas.addEventListener('pointermove', function (event) {
        event.preventDefault();
        moveDraw(event);
    });

    canvas.addEventListener('pointerup', endDraw);
    canvas.addEventListener('pointerleave', endDraw);
    canvas.addEventListener('pointercancel', endDraw);

    if (penWidthInput) {
        penWidthInput.addEventListener('input', syncPenWidth);
    }

    window.toggleSketchPad = function () {
        var pad = document.getElementById('sketch-pad');
        pad.style.display = (pad.style.display === 'none' || pad.style.display === '') ? 'block' : 'none';
    };

    window.clearCanvas = function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        fillBackground();
        syncPenWidth();
        hasStroke = false;
        if (resultBox) {
            resultBox.value = '';
        }
        if (wordLinearBox) {
            wordLinearBox.value = '';
        }
        if (historyHint) {
            historyHint.textContent = '';
        }
        currentFormulaPngUrl = '';
        currentWordMathml = '';
        currentHistoryId = null;
        currentCorrectionSuggestions = [];
        if (correctionButton) {
            correctionButton.disabled = true;
        }
    };

    window.copyWordFormula = function () {
        if (!wordLinearBox) {
            return Promise.resolve();
        }
        return copyWordMathML(currentWordMathml, wordLinearBox.value)
            .then(function () {
                if (historyHint) {
                    historyHint.textContent = 'Word 公式已复制';
                }
            })
            .catch(function (error) {
                if (String(error && error.message) !== 'empty') {
                    alert('复制 Word 公式失败，请重试。');
                }
            });
    };

    window.copySketchLatex = function () {
        if (!resultBox) {
            return Promise.resolve();
        }
        return copyPlainText(resultBox.value)
            .then(function () {
                if (historyHint) {
                    historyHint.textContent = 'LaTeX 已复制';
                }
            })
            .catch(function (error) {
                if (String(error && error.message) !== 'empty') {
                    alert('复制 LaTeX 失败，请重试。');
                }
            });
    };

    window.copySketchPng = function () {
        return copyPngFromUrl(currentFormulaPngUrl)
            .then(function () {
                if (historyHint) {
                    historyHint.textContent = 'PNG 已复制';
                }
            })
            .catch(function (error) {
                if (String(error && error.message) === 'unsupported') {
                    alert('当前浏览器暂不支持复制 PNG，请更换 Chromium 内核浏览器后再试。');
                } else if (String(error && error.message) !== 'empty') {
                    alert('复制 PNG 失败，请重试。');
                }
            });
    };

    function applyCorrectedSketchResult(data) {
        if (resultBox) {
            resultBox.value = data.latex || '';
        }
        if (wordLinearBox) {
            wordLinearBox.value = data.word_linear || '';
        }
        currentFormulaPngUrl = data.formula_png_url || '';
        currentWordMathml = data.word_mathml || '';
        currentHistoryId = data.history_id || currentHistoryId;
        currentCorrectionSuggestions = data.correction_suggestions || [];
        if (correctionButton) {
            correctionButton.disabled = !currentHistoryId;
        }
        closeSketchCorrectionDialog();
        if (historyHint) {
            historyHint.textContent = '纠错已保存，结果已更新';
        }
    }

    function renderSketchCorrectionOptions() {
        if (!correctionOptions) {
            return;
        }
        correctionOptions.innerHTML = '';
        if (!currentCorrectionSuggestions.length) {
            var empty = document.createElement('p');
            empty.textContent = '暂无推荐，请手动输入正确公式。';
            correctionOptions.appendChild(empty);
            return;
        }

        currentCorrectionSuggestions.forEach(function (item) {
            var option = document.createElement('button');
            option.type = 'button';
            option.className = 'button ghost correction-option';
            option.textContent = item.formula + (item.count ? '（使用 ' + item.count + ' 次）' : '');
            option.addEventListener('click', function () {
                submitSketchCorrection(item.formula);
            });
            correctionOptions.appendChild(option);
        });
    }

    window.openSketchCorrectionDialog = function () {
        if (!currentHistoryId) {
            alert('请先完成一次识别。');
            return;
        }
        if (correctionInput) {
            correctionInput.value = '';
        }
        renderSketchCorrectionOptions();
        if (correctionModal) {
            correctionModal.style.display = 'flex';
        }
        if (correctionInput) {
            correctionInput.focus();
        }
    };

    window.closeSketchCorrectionDialog = function () {
        if (correctionModal) {
            correctionModal.style.display = 'none';
        }
    };

    window.submitSketchManualCorrection = function () {
        submitSketchCorrection(correctionInput ? correctionInput.value : '');
    };

    function closeSketchCorrectionDialog() {
        window.closeSketchCorrectionDialog();
    }

    function submitSketchCorrection(correctText) {
        var formula = String(correctText || '').trim();
        if (!formula) {
            alert('请输入或选择正确公式。');
            return;
        }
        if (historyHint) {
            historyHint.textContent = '正在保存纠错...';
        }

        fetch('/corrections', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                history_id: currentHistoryId,
                correct_text: formula
            })
        })
            .then(function (response) {
                return response.json();
            })
            .then(function (data) {
                if (!data.success) {
                    if (historyHint) {
                        historyHint.textContent = data.error || '纠错保存失败';
                    }
                    alert(data.error || '纠错保存失败');
                    return;
                }
                applyCorrectedSketchResult(data);
            })
            .catch(function (error) {
                if (historyHint) {
                    historyHint.textContent = '纠错保存失败';
                }
                alert('纠错保存失败：' + error);
            });
    }

    window.recognizeSketch = function () {
        if (!hasStroke) {
            alert('画布为空，请先书写公式。');
            return;
        }

        if (resultBox) {
            resultBox.value = '识别中...';
        }
        currentHistoryId = null;
        currentCorrectionSuggestions = [];
        if (correctionButton) {
            correctionButton.disabled = true;
        }

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/recognize_sketch', true);
        xhr.setRequestHeader('Content-Type', 'application/json');

        xhr.onreadystatechange = function () {
            if (xhr.readyState !== 4) {
                return;
            }

            if (xhr.status === 200) {
                var data = JSON.parse(xhr.responseText);
                if (data.success) {
                    resultBox.value = data.latex;
                    if (wordLinearBox) {
                        wordLinearBox.value = data.word_linear || '';
                    }
                    currentFormulaPngUrl = data.formula_png_url || '';
                    currentWordMathml = data.word_mathml || '';
                    currentHistoryId = data.history_id || null;
                    currentCorrectionSuggestions = data.correction_suggestions || [];
                    if (correctionButton) {
                        correctionButton.disabled = !currentHistoryId;
                    }
                    if (historyHint) {
                        historyHint.textContent = data.history_url ? '本次识别已写入历史记录' : '';
                    }
                } else {
                    resultBox.value = '识别失败';
                    alert(data.error || '未知错误');
                }
            } else {
                resultBox.value = '请求失败';
                alert('识别请求失败，状态码：' + xhr.status);
            }
        };

        xhr.send(JSON.stringify({ image: canvas.toDataURL('image/png') }));
    };

    initializeCanvas();
})();
