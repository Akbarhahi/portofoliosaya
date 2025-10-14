// Chatbot Widget Logic
function chatbotWidgetInit() {
  const chatbotForm = document.getElementById('chatbot-form');
  const chatbotInput = document.getElementById('chatbot-input');
  const chatbotMessages = document.getElementById('chatbot-messages');

  function addMessage(text, sender) {
    const msg = document.createElement('div');
    msg.style.margin = '8px 0';
    msg.style.textAlign = sender === 'user' ? 'right' : 'left';
    msg.innerHTML = `<span style="display:inline-block;padding:8px 14px;border-radius:16px;max-width:80%;background:${sender==='user'?'#a883ff':'#333'};color:#fff;">${text}</span>`;
    chatbotMessages.appendChild(msg);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
  }

  chatbotForm.onsubmit = async function(e) {
    e.preventDefault();
    const userMsg = chatbotInput.value.trim();  
    if (!userMsg) return;
    addMessage(userMsg, 'user');
    chatbotInput.value = '';
    addMessage('...', 'bot');
    try {
      const res = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg })
      });
      const data = await res.json();
      chatbotMessages.lastChild.remove();
      addMessage(data.response, 'bot');
    } catch {
      chatbotMessages.lastChild.remove();
      addMessage('Gagal terhubung ke server chatbot.', 'bot');
    }
  };
}
