function submitQuestion() {
  event.preventDefault();
  const question = document.getElementById("question").value;
  fetch(`/api/answer?question=${question}`)
    .then(response => response.json())
    .then(data => {
      const answer = data.answer;
      document.getElementById("answer").innerHTML = answer;
    });
}

