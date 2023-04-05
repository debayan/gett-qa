function submitQuestion() {
  event.preventDefault();
  const question = document.getElementById("question").value;
  fetch(`http://ltgpu2:5000/answer`,{
    method: 'POST',
    headers: {
    'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      question: question
    })
  })
    .then(response => response.json())
    .then(data => {
      const jsonOutput = JSON.stringify(data, null, 2);
  const formattedOutput = `<pre>${jsonOutput}</pre>`;
  document.getElementById('answer').innerHTML = formattedOutput;
    });
}

