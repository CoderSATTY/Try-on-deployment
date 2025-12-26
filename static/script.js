let userEmail = "";

async function sendCode() {
    const name = document.getElementById("name").value;
    const email = document.getElementById("email").value;
    const msg = document.getElementById("msg-login");

    if (!email || !name) {
        msg.style.color = "#ef4444"; // Red
        msg.innerText = "Please fill in all fields.";
        return;
    }

    msg.style.color = "#fbbf24"; // Yellow
    msg.innerText = "Processing...";
    
    try {
        const response = await fetch("/api/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: email, name: name })
        });
        
        const data = await response.json();

        if (data.success) {
            userEmail = email;
            document.getElementById("step-login").style.display = "none";
            document.getElementById("step-verify").style.display = "block";
        } else {
            msg.style.color = "#ef4444";
            msg.innerText = data.message;
        }
    } catch (e) {
        msg.innerText = "Connection Error";
    }
}

async function verifyCode() {
    const code = document.getElementById("code").value;
    const msg = document.getElementById("msg-verify");

    msg.style.color = "#fbbf24";
    msg.innerText = "Verifying...";

    try {
        const response = await fetch("/api/verify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: userEmail, code: code })
        });

        const data = await response.json();

        if (data.success) {
            msg.style.color = "#22c55e"; 
            msg.innerText = "Success! Redirecting...";
            
            setTimeout(() => {
                // --- FIX IS HERE: Added '/' before '?' ---
                window.location.href = `/gradio/?user=${encodeURIComponent(userEmail)}`;
            }, 1000);
        } else {
            msg.style.color = "#ef4444";
            msg.innerText = "Incorrect code.";
        }
    } catch (e) {
        msg.innerText = "Connection Error";
    }
}