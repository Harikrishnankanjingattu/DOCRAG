/**
 * Document RAG Studio — Frontend Controller
 * Handles document ingestion, live pipeline demonstration, vector DB similarity inspection, and grounded chat.
 */

// Application State
const state = {
  currentDoc: null,
  chunks: [],
  pipelineSettings: {
    chunkSize: 400,
    chunkOverlap: 80,
    topK: 3,
    provider: 'local',
    apiKey: ''
  },
  samples: [],
  isPipelineCollapsed: false
};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const chunkSizeSlider = document.getElementById('chunkSizeSlider');
const chunkSizeDisplay = document.getElementById('chunkSizeDisplay');
const chunkOverlapSlider = document.getElementById('chunkOverlapSlider');
const chunkOverlapDisplay = document.getElementById('chunkOverlapDisplay');
const ingestProgress = document.getElementById('ingestProgress');
const ingestStatusText = document.getElementById('ingestStatusText');

const uploadSection = document.getElementById('uploadSection');
const pipelineSection = document.getElementById('pipelineSection');
const chatAndCompareSection = document.getElementById('chatAndCompareSection');

const activeDocBadge = document.getElementById('activeDocBadge');
const activeDocName = document.getElementById('activeDocName');
const btnResetDoc = document.getElementById('btnResetDoc');

// Pipeline Tabs & Stats
const pipelineDocTitle = document.getElementById('pipelineDocTitle');
const statWords = document.getElementById('statWords');
const statChunks = document.getElementById('statChunks');
const statDim = document.getElementById('statDim');
const statModel = document.getElementById('statModel');
const tabChunksCount = document.getElementById('tabChunksCount');
const chunksCardGrid = document.getElementById('chunksCardGrid');
const embeddingsPreviewGrid = document.getElementById('embeddingsPreviewGrid');
const vectorTableBody = document.getElementById('vectorTableBody');
const btnTogglePipelineCollapse = document.getElementById('btnTogglePipelineCollapse');
const pipelineTabContent = document.getElementById('pipelineTabContent');

// Chat & Comparison
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const btnClearChat = document.getElementById('btnClearChat');
const samplePromptsList = document.getElementById('samplePromptsList');

const compareStatusBadge = document.getElementById('compareStatusBadge');
const queryVectorStrip = document.getElementById('queryVectorStrip');
const queryDimBadge = document.getElementById('queryDimBadge');
const queryVectorFloats = document.getElementById('queryVectorFloats');
const comparisonEmptyState = document.getElementById('comparisonEmptyState');
const comparisonList = document.getElementById('comparisonList');

// Modals
const settingsModal = document.getElementById('settingsModal');
const btnOpenSettings = document.getElementById('btnOpenSettings');
const btnCloseSettings = document.getElementById('btnCloseSettings');
const btnSaveSettings = document.getElementById('btnSaveSettings');
const topKSlider = document.getElementById('topKSlider');
const topKDisplay = document.getElementById('topKDisplay');
const providerSelect = document.getElementById('providerSelect');
const apiKeyField = document.getElementById('apiKeyField');
const apiKeyLabel = document.getElementById('apiKeyLabel');
const apiKeyInput = document.getElementById('apiKeyInput');

const chunkDetailModal = document.getElementById('chunkDetailModal');
const modalChunkBadge = document.getElementById('modalChunkBadge');
const modalChunkMeta = document.getElementById('modalChunkMeta');
const modalChunkText = document.getElementById('modalChunkText');
const modalVectorFloats = document.getElementById('modalVectorFloats');
const btnCloseChunkModal = document.getElementById('btnCloseChunkModal');
const btnCloseChunkModalBtn = document.getElementById('btnCloseChunkModalBtn');

// Initialize App (Starts clean, waiting for user document upload)
document.addEventListener('DOMContentLoaded', () => {
  initEventListeners();
});

// Event Listeners
function initEventListeners() {
  // Sliders
  chunkSizeSlider.addEventListener('input', (e) => {
    state.pipelineSettings.chunkSize = parseInt(e.target.value);
    chunkSizeDisplay.textContent = e.target.value;
  });

  chunkOverlapSlider.addEventListener('input', (e) => {
    state.pipelineSettings.chunkOverlap = parseInt(e.target.value);
    chunkOverlapDisplay.textContent = e.target.value;
  });

  topKSlider.addEventListener('input', (e) => {
    state.pipelineSettings.topK = parseInt(e.target.value);
    topKDisplay.textContent = e.target.value;
  });

  // Drag and Drop
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-black', 'bg-zinc-100');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-black', 'bg-zinc-100');
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-black', 'bg-zinc-100');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Pipeline Tabs
  const tabBtns = document.querySelectorAll('#pipelineTabs .tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      document.getElementById(targetId).classList.remove('hidden');
    });
  });

  // Collapse Pipeline
  btnTogglePipelineCollapse.addEventListener('click', () => {
    state.isPipelineCollapsed = !state.isPipelineCollapsed;
    if (state.isPipelineCollapsed) {
      pipelineTabContent.classList.add('hidden');
      btnTogglePipelineCollapse.textContent = '[ Expand Details ]';
    } else {
      pipelineTabContent.classList.remove('hidden');
      btnTogglePipelineCollapse.textContent = '[ Collapse Details ]';
    }
  });

  // Chat Form
  chatForm.addEventListener('submit', handleSendMessage);
  btnClearChat.addEventListener('click', clearChatThread);

  // Reset Document
  btnResetDoc.addEventListener('click', resetDocument);

  // Settings Modal
  btnOpenSettings.addEventListener('click', () => {
    settingsModal.classList.remove('hidden');
  });
  btnCloseSettings.addEventListener('click', () => {
    settingsModal.classList.add('hidden');
  });
  btnSaveSettings.addEventListener('click', () => {
    state.pipelineSettings.provider = providerSelect.value;
    state.pipelineSettings.apiKey = apiKeyInput.value.trim();
    settingsModal.classList.add('hidden');
  });
  providerSelect.addEventListener('change', (e) => {
    if (e.target.value === 'local') {
      apiKeyField.classList.add('hidden');
    } else {
      apiKeyField.classList.remove('hidden');
      apiKeyLabel.textContent = e.target.value === 'gemini' ? 'Gemini API Key:' : 'OpenAI API Key:';
    }
  });

  // Chunk Modal Close
  btnCloseChunkModal.addEventListener('click', () => chunkDetailModal.classList.add('hidden'));
  btnCloseChunkModalBtn.addEventListener('click', () => chunkDetailModal.classList.add('hidden'));
}

// Upload Custom File
async function handleFileUpload(file) {
  showIngestProgress(`Parsing ${file.name}, chunking text, and building vector database...`);
  const formData = new FormData();
  formData.append('file', file);
  formData.append('chunk_size', state.pipelineSettings.chunkSize);
  formData.append('chunk_overlap', state.pipelineSettings.chunkOverlap);
  if (state.pipelineSettings.provider !== 'local') {
    formData.append('api_provider', state.pipelineSettings.provider);
    formData.append('api_key', state.pipelineSettings.apiKey);
  }

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(errData.detail || 'Upload failed');
    }
    const data = await res.json();
    onIngestionSuccess(data);
  } catch (err) {
    alert('Upload error: ' + err.message);
  } finally {
    hideIngestProgress();
  }
}

function showIngestProgress(msg) {
  ingestStatusText.textContent = msg;
  ingestProgress.classList.remove('hidden');
}

function hideIngestProgress() {
  ingestProgress.classList.add('hidden');
}

// Ingestion Success Handler
function onIngestionSuccess(data) {
  state.currentDoc = data.document;
  state.chunks = data.chunks;

  // Update UI Elements
  activeDocBadge.classList.remove('hidden');
  activeDocName.textContent = data.document.name;
  btnResetDoc.classList.remove('hidden');

  pipelineDocTitle.textContent = data.document.name;
  statWords.textContent = data.document.total_words.toLocaleString();
  statChunks.textContent = data.document.total_chunks;
  statDim.textContent = `${data.document.embedding_dimension}-D`;
  statModel.textContent = data.document.embedding_model.split(' ')[0];
  tabChunksCount.textContent = data.document.total_chunks;

  // Render Pipeline Demonstrations
  renderChunkCards(data.chunks);
  renderEmbeddingsPreview(data.chunks);
  renderVectorDatabaseTable(data.chunks);

  // Setup Sample Prompts for active doc
  setupSamplePrompts(data.document.name);

  // Transition: collapse upload section, display pipeline & chat sections
  uploadSection.classList.add('hidden');
  pipelineSection.classList.remove('hidden');
  chatAndCompareSection.classList.remove('hidden');

  // Reset comparison panel
  comparisonList.innerHTML = '';
  comparisonList.classList.add('hidden');
  comparisonEmptyState.classList.remove('hidden');
  queryVectorStrip.classList.add('hidden');
  compareStatusBadge.textContent = 'Ready for query';

  // Scroll smoothly to chat section
  chatAndCompareSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Render Chunking Breakdown Cards
function renderChunkCards(chunks) {
  chunksCardGrid.innerHTML = '';
  chunks.forEach(chunk => {
    const card = document.createElement('div');
    card.className = 'border border-zinc-200 hover:border-black rounded-lg p-3 bg-white transition flex flex-col justify-between shadow-xs';
    
    // Highlight overlap boundaries in text snippet
    let displayText = escapeHtml(chunk.text);
    if (chunk.overlap_start && displayText.startsWith(escapeHtml(chunk.overlap_start))) {
      const overlapLen = escapeHtml(chunk.overlap_start).length;
      displayText = `<span class="overlap-boundary" title="Preserved from previous chunk">${displayText.substring(0, overlapLen)}</span>` + displayText.substring(overlapLen);
    }

    card.innerHTML = `
      <div>
        <div class="flex items-center justify-between font-mono text-[11px] mb-2 pb-1.5 border-b border-zinc-100">
          <span class="font-bold bg-black text-white px-1.5 py-0.5 rounded text-[10px]">Chunk #${chunk.index}</span>
          <span class="text-zinc-500">${chunk.token_count} tokens | chars: ${chunk.char_start}-${chunk.char_end}</span>
        </div>
        <div class="chunk-text-wrapper">
          ${displayText}
        </div>
      </div>
      <div class="mt-3 pt-2 border-t border-zinc-100 flex items-center justify-between text-[11px] font-mono">
        <span class="text-zinc-400">ID: ${chunk.id}</span>
        <button class="text-black font-semibold hover:underline" data-view-chunk="${chunk.index}">
          Inspect Vector &rarr;
        </button>
      </div>
    `;

    card.querySelector(`[data-view-chunk="${chunk.index}"]`).addEventListener('click', () => {
      openChunkDetailModal(chunk);
    });

    chunksCardGrid.appendChild(card);
  });
}

// Render Embeddings Preview
function renderEmbeddingsPreview(chunks) {
  embeddingsPreviewGrid.innerHTML = '';
  chunks.slice(0, 8).forEach(chunk => {
    const card = document.createElement('div');
    card.className = 'border border-zinc-200 rounded-lg p-3.5 bg-white shadow-xs font-mono text-xs';
    
    const previewFloats = chunk.vector_preview ? chunk.vector_preview.map(v => (v >= 0 ? '+' : '') + v.toFixed(4)).join(', ') : '';
    
    // Sparkline bars
    let sparkHtml = '';
    if (chunk.vector_spark) {
      sparkHtml = chunk.vector_spark.map(val => {
        const height = Math.max(3, Math.min(20, Math.round(val * 40)));
        return `<div class="sparkbar" style="height: ${height}px;" title="Magnitude: ${val}"></div>`;
      }).join('');
    }

    card.innerHTML = `
      <div class="flex items-center justify-between mb-2">
        <span class="font-bold text-zinc-900">Chunk #${chunk.index} Embedding</span>
        <span class="text-[10px] text-zinc-500">Dim: ${chunk.vector_dim || 128} | L2 Norm: ${chunk.vector_norm || '1.0000'}</span>
      </div>
      <div class="mb-2">
        <span class="text-[10px] text-zinc-400 block mb-1">Normalized Vector Sample (First 12 components):</span>
        <div class="p-2 bg-zinc-50 border border-zinc-200 rounded text-[10px] text-zinc-700 overflow-x-auto whitespace-nowrap">
          [${previewFloats}, ...]
        </div>
      </div>
      <div class="flex items-center justify-between text-[10px] text-zinc-500 pt-1">
        <span>Magnitude Spectrum:</span>
        <div class="sparkbar-container w-28">
          ${sparkHtml}
        </div>
      </div>
    `;
    embeddingsPreviewGrid.appendChild(card);
  });
}

// Render Vector Database Table
function renderVectorDatabaseTable(chunks) {
  vectorTableBody.innerHTML = '';
  chunks.forEach(chunk => {
    const row = document.createElement('tr');
    row.className = 'hover:bg-zinc-50 transition cursor-pointer';
    const previewFloats = chunk.vector_preview ? chunk.vector_preview.slice(0, 8).map(v => (v >= 0 ? '+' : '') + v.toFixed(3)).join(', ') : 'N/A';
    const textSnippet = chunk.text.length > 50 ? chunk.text.substring(0, 50) + '...' : chunk.text;

    row.innerHTML = `
      <td class="p-2.5 font-bold">${chunk.id}</td>
      <td class="p-2.5 text-zinc-500">${chunk.char_start} - ${chunk.char_end}</td>
      <td class="p-2.5 text-zinc-600">${chunk.token_count}</td>
      <td class="p-2.5 text-zinc-600">${chunk.vector_norm || '1.000'}</td>
      <td class="p-2.5 text-zinc-700 truncate max-w-xs font-mono text-[10px]">[${previewFloats}]</td>
      <td class="p-2.5 text-zinc-600 font-sans truncate max-w-xs">${escapeHtml(textSnippet)}</td>
    `;
    row.addEventListener('click', () => openChunkDetailModal(chunk));
    vectorTableBody.appendChild(row);
  });
}

// Dynamic Prompt Suggestions based strictly on the uploaded document content
function setupSamplePrompts(docName) {
  samplePromptsList.innerHTML = '';
  let prompts = [
    "Summarize the main topic of this document.",
    "What are the key technical concepts explained in this text?"
  ];

  // Dynamically extract headings or key subjects from the user's uploaded chunks
  if (state.chunks && state.chunks.length > 0) {
    const extractedTopics = [];
    state.chunks.forEach(chunk => {
      // Find Markdown headings or capitalized title lines
      const lines = chunk.text.split('\n');
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('#') || trimmed.startsWith('##') || trimmed.startsWith('###')) {
          const cleanHeading = trimmed.replace(/^#+\s*/, '').replace(/[:.]*$/, '').trim();
          if (cleanHeading.length > 4 && cleanHeading.length < 50 && !extractedTopics.includes(cleanHeading)) {
            extractedTopics.push(cleanHeading);
          }
        }
      }
    });

    // Add up to 3 dynamic questions based on detected topics from the user's document
    extractedTopics.slice(0, 3).forEach(topic => {
      prompts.push(`Explain: "${topic}"`);
    });
  }

  prompts.slice(0, 4).forEach(p => {
    const chip = document.createElement('button');
    chip.className = 'px-2 py-1 bg-white hover:bg-black hover:text-white border border-zinc-200 rounded text-zinc-700 whitespace-nowrap transition cursor-pointer';
    chip.textContent = p;
    chip.addEventListener('click', () => {
      chatInput.value = p;
      handleSendMessage(new Event('submit'));
    });
    samplePromptsList.appendChild(chip);
  });
}

// Send Message Handler
async function handleSendMessage(e) {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  // Add User Message to Chat
  appendUserMessage(query);
  chatInput.value = '';

  // Update Status & UI
  compareStatusBadge.textContent = 'Ranking DB Vectors...';
  compareStatusBadge.classList.add('bg-black', 'text-white');

  // Add Bot Loading Bubble
  const loadingBubble = appendBotLoadingBubble();

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        top_k: state.pipelineSettings.topK,
        api_provider: state.pipelineSettings.provider,
        api_key: state.pipelineSettings.apiKey
      })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: 'Query failed' }));
      throw new Error(errData.detail || 'Query failed');
    }

    const data = await res.json();

    // Remove loading bubble and append actual answer
    loadingBubble.remove();
    appendBotMessage(data);

    // Render Real-time Vector Comparison Demonstration
    renderVectorComparisons(data);

  } catch (err) {
    loadingBubble.remove();
    appendBotErrorMessage('Error: ' + err.message);
  } finally {
    compareStatusBadge.textContent = 'Query completed';
    compareStatusBadge.classList.remove('bg-black', 'text-white');
  }
}

// Render Real-time Vector Similarity Comparison
function renderVectorComparisons(data) {
  comparisonEmptyState.classList.add('hidden');
  comparisonList.classList.remove('hidden');
  queryVectorStrip.classList.remove('hidden');

  // Render Query Vector Preview
  queryDimBadge.textContent = `${data.query_vector_dim}-D Unit Vector`;
  const qFloats = data.query_vector_preview.map(v => (v >= 0 ? '+' : '') + v.toFixed(4)).join(', ');
  queryVectorFloats.textContent = `[${qFloats}, ...]`;

  // Render Comparisons Ranking
  comparisonList.innerHTML = '';
  data.all_comparisons.forEach(item => {
    const row = document.createElement('div');
    const isTopK = item.is_top_k;
    
    row.className = `p-3 rounded-lg border transition cursor-pointer ${
      isTopK 
        ? 'border-black bg-white shadow-xs' 
        : 'border-zinc-200 bg-zinc-50/70 hover:bg-zinc-100/80 opacity-75'
    }`;

    const scorePct = item.percentage;
    const rawCos = item.cosine_similarity;

    row.innerHTML = `
      <div class="flex items-center justify-between text-xs font-mono">
        <div class="flex items-center space-x-2">
          <span class="font-bold px-1.5 py-0.5 rounded text-[10px] ${isTopK ? 'bg-black text-white' : 'bg-zinc-200 text-zinc-700'}">
            Rank #${item.rank}
          </span>
          <span class="font-semibold text-zinc-900">Chunk #${item.chunk_index}</span>
          ${isTopK ? '<span class="text-[9px] bg-emerald-100 text-emerald-800 font-bold px-1.5 py-0.5 rounded">RETRIEVED</span>' : '<span class="text-[9px] text-zinc-400">FILTERED</span>'}
        </div>
        <div class="text-right">
          <span class="font-bold text-xs ${isTopK ? 'text-black' : 'text-zinc-500'}">${scorePct}%</span>
          <span class="text-[10px] text-zinc-400 block font-mono">cos: ${rawCos}</span>
        </div>
      </div>

      <!-- Similarity Progress Bar -->
      <div class="sim-bar-bg">
        <div class="sim-bar-fill ${isTopK ? '' : 'filtered'}" style="width: ${Math.max(4, scorePct)}%;"></div>
      </div>

      <!-- Text snippet preview -->
      <p class="mt-2 text-[11px] text-zinc-600 line-clamp-2 font-sans leading-relaxed">
        ${escapeHtml(item.text_snippet)}
      </p>
    `;

    // Click to view full chunk details
    row.addEventListener('click', () => {
      const fullChunk = state.chunks.find(c => c.index === item.chunk_index);
      if (fullChunk) openChunkDetailModal(fullChunk);
    });

    comparisonList.appendChild(row);
  });
}

// Chat UI Appends
function appendUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'flex items-start space-x-2 justify-end';
  msg.innerHTML = `
    <div class="msg-user px-3.5 py-2.5 text-xs max-w-[85%] leading-relaxed">
      ${escapeHtml(text)}
    </div>
  `;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendBotLoadingBubble() {
  const msg = document.createElement('div');
  msg.className = 'flex items-start space-x-3';
  msg.innerHTML = `
    <div class="w-6 h-6 rounded bg-black text-white flex items-center justify-center font-mono text-[10px] shrink-0 font-bold">
      AI
    </div>
    <div class="bg-zinc-50 border border-zinc-200 rounded-lg p-3 text-xs leading-relaxed max-w-[85%] flex items-center space-x-2">
      <div class="spinner w-3.5 h-3.5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
      <span class="font-mono text-zinc-600 text-[11px]">Retrieving top vectors & synthesizing answer...</span>
    </div>
  `;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return msg;
}

function appendBotMessage(data) {
  const msg = document.createElement('div');
  msg.className = 'flex items-start space-x-3';

  // Format citations into interactive badges
  let citationsHtml = '';
  if (data.citations && data.citations.length > 0) {
    citationsHtml = '<div class="mt-3 pt-2.5 border-t border-zinc-200/80 flex flex-wrap items-center gap-1.5 text-[11px] font-mono"><span class="text-zinc-500 mr-1">Sources Cited:</span>';
    data.citations.forEach(c => {
      citationsHtml += `
        <button class="citation-badge" data-chunk-idx="${c.chunk_index}">
          <span>Chunk #${c.chunk_index}</span>
          <span class="text-zinc-500 font-normal">(${c.percentage}%)</span>
        </button>
      `;
    });
    citationsHtml += '</div>';
  }

  // Parse simple markdown formatting (bold, lists)
  let formattedText = escapeHtml(data.answer)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n- /g, '<br>• ');

  msg.innerHTML = `
    <div class="w-6 h-6 rounded bg-black text-white flex items-center justify-center font-mono text-[10px] shrink-0 font-bold mt-1">
      AI
    </div>
    <div class="msg-bot p-4 text-xs leading-relaxed max-w-[90%] shadow-xs">
      <div class="text-zinc-900 leading-relaxed font-sans">
        ${formattedText}
      </div>
      ${citationsHtml}
    </div>
  `;

  // Attach citation badge clicks
  msg.querySelectorAll('.citation-badge').forEach(badge => {
    badge.addEventListener('click', () => {
      const idx = parseInt(badge.getAttribute('data-chunk-idx'));
      const chunk = state.chunks.find(c => c.index === idx);
      if (chunk) openChunkDetailModal(chunk);
    });
  });

  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendBotErrorMessage(errText) {
  const msg = document.createElement('div');
  msg.className = 'flex items-start space-x-3';
  msg.innerHTML = `
    <div class="w-6 h-6 rounded bg-red-600 text-white flex items-center justify-center font-mono text-[10px] shrink-0 font-bold">
      !
    </div>
    <div class="bg-red-50 border border-red-200 rounded-lg p-3 text-xs text-red-700 leading-relaxed max-w-[85%]">
      ${escapeHtml(errText)}
    </div>
  `;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearChatThread() {
  chatMessages.innerHTML = `
    <div class="flex items-start space-x-3">
      <div class="w-6 h-6 rounded bg-black text-white flex items-center justify-center font-mono text-[10px] shrink-0 font-bold">
        AI
      </div>
      <div class="bg-zinc-50 border border-zinc-200 rounded-lg p-3 text-xs leading-relaxed max-w-[88%]">
        <p class="font-semibold text-zinc-900 mb-1">Chat Thread Reset</p>
        <p class="text-zinc-700">Ask a question to see the vector similarity comparison in action.</p>
      </div>
    </div>
  `;
}

// Reset Document Handler
async function resetDocument() {
  try {
    await fetch('/api/reset', { method: 'POST' });
  } catch (e) {
    console.error(e);
  }
  state.currentDoc = null;
  state.chunks = [];

  activeDocBadge.classList.add('hidden');
  btnResetDoc.classList.add('hidden');
  pipelineSection.classList.add('hidden');
  chatAndCompareSection.classList.add('hidden');
  uploadSection.classList.remove('hidden');
  clearChatThread();
  fileInput.value = '';
}

// Chunk Details Modal
function openChunkDetailModal(chunk) {
  modalChunkBadge.textContent = `Chunk #${chunk.index} (${chunk.id})`;
  modalChunkMeta.textContent = `Tokens: ${chunk.token_count} | Characters: ${chunk.char_start} - ${chunk.char_end}`;
  modalChunkText.textContent = chunk.text;

  // Format full vector embedding preview
  if (chunk.embedding) {
    const fullArray = chunk.embedding.map((val, i) => `dim[${i}]: ${(val >= 0 ? '+' : '') + val.toFixed(5)}`).join('  |  ');
    modalVectorFloats.textContent = fullArray;
  } else if (chunk.vector_preview) {
    const sampleArray = chunk.vector_preview.map((val, i) => `dim[${i}]: ${(val >= 0 ? '+' : '') + val.toFixed(4)}`).join('  |  ');
    modalVectorFloats.textContent = sampleArray + ' ...';
  } else {
    modalVectorFloats.textContent = 'Vector embedding calculated in memory.';
  }

  chunkDetailModal.classList.remove('hidden');
}

// Check existing document on server
async function checkExistingDocument() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.has_document && data.indexed_chunks > 0) {
      const chunksRes = await fetch('/api/chunks');
      const chunksData = await chunksRes.json();
      onIngestionSuccess({
        document: {
          name: chunksData.document_name,
          type: 'txt',
          total_chars: 0,
          total_words: 0,
          total_chunks: chunksData.total_chunks,
          chunk_size: 400,
          chunk_overlap: 80,
          embedding_model: 'Local Dense Vector Space',
          embedding_dimension: 128
        },
        chunks: chunksData.chunks
      });
    }
  } catch (e) {
    // Fresh start
  }
}

// Utility: Escape HTML
function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>'"]/g, tag => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[tag] || tag));
}
