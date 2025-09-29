document.addEventListener('DOMContentLoaded', function () {
  // Initialize mermaid
  mermaid.initialize({ startOnLoad: true });

  // Get DOM elements
  const form = document.getElementById('uploadForm');
  const videoInput = document.getElementById('videoFile');
  const results = document.getElementById('results');
  const processingMsg = document.getElementById('processingMessage');
  const transcriptEl = document.getElementById('transcript');
  const summaryAbsEl = document.getElementById('summary-abstractive');
  const summaryExtEl = document.getElementById('summary-extractive');
  const generateNotesBtn = document.getElementById('generate-notes');
  const notesContainer = document.getElementById('notes-container');
  const generateFlowchartBtn = document.getElementById('generateFlowchartBtn') || document.getElementById('generate-flowchart');
  const flowchartContainer = document.getElementById('flowchart');

  let noteId = 1;
  let currentVideoData = null;

  // Ensure progress is visible before the form is submitted
  // if (progress) {
  //   progress.classList.remove('hidden');
  //   progress.style.display = '';
  // }

  // File input handling
  const customFileBtn = document.getElementById('customFileBtn');
  const fileNameDisplay = document.getElementById('fileNameDisplay');
  if (customFileBtn && videoInput && fileNameDisplay) {
    customFileBtn.addEventListener('click', function (e) {
      e.preventDefault();
      videoInput.click();
    });

    videoInput.addEventListener('change', function () {
      const file = videoInput.files[0];
      fileNameDisplay.textContent = file ? file.name : 'No file chosen';
    });
  }

  // Input type switch
  document.querySelectorAll('input[name="inputType"]').forEach(input => {
    input.addEventListener('change', (e) => {
      const fileInput = document.getElementById('fileInput');
      const urlInput = document.getElementById('urlInput');

      if (e.target.value === 'file') {
        fileInput.style.display = 'block';
        urlInput.style.display = 'none';
        videoInput.required = true;
        document.getElementById('youtubeUrl').required = false;
      } else {
        fileInput.style.display = 'none';
        urlInput.style.display = 'block';
        videoInput.required = false;
        document.getElementById('youtubeUrl').required = true;
      }
    });
  });

  // Handle form submission
  form.addEventListener('submit', async function (event) {
    event.preventDefault();


    results.classList.add('hidden');
    processingMsg.classList.remove('hidden');

    const inputType = document.querySelector('input[name="inputType"]:checked')?.value;
    const errorDiv = document.getElementById('error-message');
    if (transcriptEl) transcriptEl.textContent = '';
    if (summaryAbsEl) summaryAbsEl.textContent = '';
    if (summaryExtEl) summaryExtEl.textContent = '';
    errorDiv.classList.add('hidden');

    try {
      let formData = new FormData();
      let endpoint = '';

      if (inputType === 'file') {
        const videoFile = videoInput.files[0];
        if (!videoFile) throw new Error('Please select a video file');
        formData.append('video', videoFile);
        endpoint = '/process-video';
      } else {
        const youtubeUrl = document.getElementById('youtubeUrl').value;
        if (!youtubeUrl) throw new Error('Please enter a YouTube URL');
        if (!youtubeUrl.match(/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/)) {
          throw new Error('Please enter a valid YouTube URL');
        }
        formData.append('url', youtubeUrl);
        endpoint = '/process-youtube';
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      currentVideoData = data;

      transcriptEl.innerHTML = (data.transcript || 'No transcript available').replace(/\n/g, '<br>');
      summaryAbsEl.innerHTML = (data.summary_abstractive || 'No abstractive summary available').replace(/\n/g, '<br>');
      summaryExtEl.textContent = data.summary_extractive || 'No extractive summary available';

      results.classList.remove('hidden');

      // Enable "Generate Notes" only if available
      generateNotesBtn.disabled = !data.structured_notes;

    } catch (error) {
      console.error(error);
      errorDiv.classList.remove('hidden');
      errorDiv.textContent = error.message;
    } finally {
      processingMsg.classList.add('hidden');
    }
  });

  // Notes
  generateNotesBtn.addEventListener('click', () => {
    if (currentVideoData && currentVideoData.structured_notes) {
      displayStructuredNotes(currentVideoData.structured_notes);
      generateNotesBtn.disabled = true;
    } else {
      alert('Please process a video first.');
    }
  });

  function displayStructuredNotes(notes) {
    notesContainer.innerHTML = '';
    if (notes && notes.headings) {
      notes.headings.forEach(heading => {
        const headingDiv = document.createElement('div');
        headingDiv.className = 'note-item';
        const headingContent = document.createElement('div');
        headingContent.className = 'note-heading highlight-heading';
        headingContent.textContent = heading;
        headingDiv.appendChild(headingContent);

        const ul = document.createElement('ul');
        ul.className = 'note-points-list';
        notes.points[heading].forEach(point => {
          const li = document.createElement('li');
          li.className = 'note-point';
          li.textContent = point;
          ul.appendChild(li);
        });
        headingDiv.appendChild(ul);
        notesContainer.appendChild(headingDiv);
      });
    }
  }

  // Flowchart generation
  if (generateFlowchartBtn) {
    generateFlowchartBtn.addEventListener('click', () => {
      if (currentVideoData?.summary_abstractive) {
        generateFlowchart(currentVideoData.summary_abstractive);
      } else {
        alert('Please upload and process a video first.');
      }
    });
  }

  function generateFlowchart(text) {
    const sentences = text.split(/[.!?]\s+/).map(s => s.trim()).filter(Boolean);
    let flowchartDef = 'graph TD\n';
    sentences.forEach((sentence, index) => {
      const cleanSentence = sentence.replace(/["\[\]]/g, '').replace(/\n/g, ' ');
      flowchartDef += `    N${index}["${cleanSentence}"]\n`;
      if (index > 0) flowchartDef += `    N${index - 1} --> N${index}\n`;
    });
    flowchartContainer.innerHTML = `<pre class="mermaid">${flowchartDef}</pre>`;
    mermaid.init(undefined, '.mermaid');
  }
});
