// Operational Dashboard Controller for Switchboard

document.addEventListener("DOMContentLoaded", () => {
    // Initialize Lucide Icons on load
    if (typeof lucide !== "undefined") {
        lucide.createIcons();
    }

    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatFeed = document.getElementById("chatFeed");
    const chatLoading = document.getElementById("chatLoading");

    // UI elements to update state
    const bankCard = document.getElementById("bankCard");
    const cardStatus = document.getElementById("cardStatus");
    const balanceAmount = document.getElementById("balanceAmount");
    const transactionList = document.getElementById("transactionList");

    // Presets elements
    const presetsRow = document.getElementById("presetsRow");

    // Nav bar Ollama status elements
    const ollamaIcon = document.getElementById("ollamaIcon");
    const ollamaStatusLabel = document.getElementById("ollamaStatusLabel");

    // Diagnostic elements
    const intentRankList = document.getElementById("intentRankList");
    const lblAction = document.getElementById("lblAction");
    const lblParamsTable = document.getElementById("lblParamsTable");
    const lblToolOutput = document.getElementById("lblToolOutput");

    // Pipeline Graph Nodes elements
    const graphMLLabel = document.getElementById("graphMLLabel");
    const graphUserQuery = document.getElementById("graphUserQuery");
    const graphMLIntent = document.getElementById("graphMLIntent");
    const graphPromptContext = document.getElementById("graphPromptContext");
    const graphAgentAction = document.getElementById("graphAgentAction");
    const graphDBUpdate = document.getElementById("graphDBUpdate");

    // Global metadata variable
    let systemMetadata = null;

    // Interactive Tab Switching
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");

    const switchTab = (tabName) => {
        tabBtns.forEach(btn => {
            if (btn.dataset.tab === tabName) {
                tabBtns.forEach(b => b.classList.remove("active"));
                tabPanels.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                document.getElementById(`tab-${tabName}`).classList.add("active");
            }
        });
    };

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            switchTab(btn.dataset.tab);
        });
    });

    // Interactive 3D Card Hover Tilt effect
    if (bankCard) {
        bankCard.addEventListener("mousemove", (e) => {
            const rect = bankCard.getBoundingClientRect();
            const x = e.clientX - rect.left - rect.width / 2;
            const y = e.clientY - rect.top - rect.height / 2;
            // Rotate card slightly based on mouse offset to create a premium 3D tilt effect
            bankCard.style.transform = `rotateY(${x / 8}deg) rotateX(${-y / 8}deg) scale(1.03)`;
        });
        bankCard.addEventListener("mouseleave", () => {
            bankCard.style.transform = "rotateY(0deg) rotateX(0deg) scale(1)";
        });
    }

    // Dynamic Metadata Fetch (Accuracy, Dataset size, classes, embedding model name)
    const loadMetadata = async () => {
        try {
            const res = await fetch("/metadata");
            if (!res.ok) throw new Error("Failed to load metadata");
            systemMetadata = await res.json();

            // Dynamic updates to Telemetry Card text
            const explanationTxt = document.querySelector("#cardML .explanation-text");
            if (explanationTxt) {
                explanationTxt.innerHTML = `
                    Dataset: <strong>${systemMetadata.dataset}</strong> (${systemMetadata.n_classes} classes, ${systemMetadata.n_train.toLocaleString()} train items). 
                    Accuracy: <strong>${(systemMetadata.accuracy * 100).toFixed(1)}%</strong>. 
                    Embeddings: <code>${systemMetadata.embedding_model.split("/")[1]}</code>.
                `;
            }

            // Dynamic updates to Pipeline Graph labels
            if (graphMLLabel) {
                graphMLLabel.textContent = `1. Intent Classifier (${systemMetadata.dataset.toUpperCase()})`;
            }

        } catch (err) {
            console.error("Metadata load error:", err);
        }
    };

    // Dynamic Presets Fetch
    const loadDynamicPresets = async () => {
        try {
            const res = await fetch("/presets");
            if (!res.ok) throw new Error("Failed to load presets");
            const data = await res.json();

            presetsRow.innerHTML = ""; // Clear loader
            data.forEach(preset => {
                const btn = document.createElement("button");
                btn.className = "preset-btn";
                btn.onclick = () => sendPreset(preset.message);

                // Dynamically assign correct vector icon based on expected tool
                let iconName = "help-circle";
                let labelText = "General Policy";

                if (preset.expected === "check_balance") {
                    iconName = "wallet";
                    labelText = "Check Balance";
                } else if (preset.expected === "freeze_card") {
                    iconName = "shield-alert";
                    labelText = "Freeze Card";
                } else if (preset.expected === "transaction_history") {
                    iconName = "receipt";
                    labelText = "Transactions";
                } else if (preset.expected === "escalate_to_human") {
                    iconName = "alert-triangle";
                    labelText = "Human Escalate";
                }

                btn.innerHTML = `
                    <i data-lucide="${iconName}" class="preset-icon"></i>
                    <span class="preset-label">${labelText}</span>
                `;
                presetsRow.appendChild(btn);
            });

            // Re-render Lucide SVG elements for dynamically added tags
            if (typeof lucide !== "undefined") {
                lucide.createIcons();
            }

        } catch (err) {
            console.error("Presets loading error:", err);
            presetsRow.innerHTML = "<span class='italic' style='font-size: 0.72rem; color: var(--text-muted);'>Presets failed to load.</span>";
        }
    };

    // Preset query helper
    window.sendPreset = (text) => {
        chatInput.value = text;
        chatForm.dispatchEvent(new Event("submit"));
    };

    // Render helper for chat messages
    const appendMessage = (sender, text) => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${sender}-msg`;
        
        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";
        
        const label = sender === "user" ? "You" : "Switchboard Agent";
        bubble.innerHTML = `<strong>${label}:</strong> ${text}`;
        
        msgDiv.appendChild(bubble);
        chatFeed.appendChild(msgDiv);
        
        // Auto-scroll
        chatFeed.parentNode.scrollTop = chatFeed.parentNode.scrollHeight;
    };

    // Flash highlights on simulated DB changes
    const flashElement = (elem, cssClass) => {
        elem.classList.add(cssClass);
        setTimeout(() => elem.classList.remove(cssClass), 1000);
    };

    // Fetch live backend state and update the UI dynamically
    const refreshDBState = async (flashType = null) => {
        try {
            const res = await fetch("/state");
            if (!res.ok) throw new Error("Failed to fetch state");
            const data = await res.json();

            // 1. Update Account Balance
            const account = data.accounts["ACC-1001"];
            if (account) {
                const formattedBalance = new Intl.NumberFormat('en-GB', {
                    style: 'currency',
                    currency: account.currency
                }).format(account.balance);
                balanceAmount.textContent = formattedBalance;
                
                // Update owner dynamically from backend state
                const cardholderVal = document.querySelector(".holder-details .value");
                if (cardholderVal) {
                    cardholderVal.textContent = account.owner.toUpperCase();
                }
                
                if (flashType === "balance") {
                    flashElement(balanceAmount, "positive-flash");
                }
            }

            // 2. Update Card Status
            const card = data.cards["CARD-4521"];
            if (card) {
                const status = card.status;
                const formattedStatus = status.charAt(0).toUpperCase() + status.slice(1);
                cardStatus.textContent = formattedStatus;
                
                if (status === "active") {
                    bankCard.className = "debit-card active";
                } else if (status === "frozen") {
                    bankCard.className = "debit-card frozen";
                }
            }

            // 3. Update Transactions Log
            if (data.transactions) {
                transactionList.innerHTML = "";
                data.transactions.forEach(tx => {
                    const tr = document.createElement("tr");
                    const isPositive = tx.amount > 0;
                    const amountClass = isPositive ? "positive" : "negative";
                    const amountSign = isPositive ? "+" : "";
                    
                    tr.innerHTML = `
                        <td class="merchant-name">${tx.merchant}</td>
                        <td class="amount ${amountClass}">${amountSign}£${Math.abs(tx.amount).toFixed(2)}</td>
                    `;
                    
                    transactionList.appendChild(tr);
                    
                    if (flashType === "transactions") {
                        flashElement(tr, "tx-flash");
                    }
                });
            }

        } catch (error) {
            console.error("Error refreshing database state:", error);
        }
    };

    // Load initial state from backend database, metadata, and dynamic presets
    refreshDBState();
    loadMetadata();
    loadDynamicPresets();

    // Chat API Request handler
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const messageText = chatInput.value.trim();
        if (!messageText) return;

        chatInput.value = "";
        appendMessage("user", messageText);

        // Show thinking animation & update Ollama status in Navbar
        chatLoading.style.display = "flex";
        if (ollamaIcon) {
            ollamaIcon.style.color = "var(--color-yellow)";
            ollamaIcon.classList.add("pulse");
        }
        ollamaStatusLabel.textContent = "Reasoning...";
        chatFeed.parentNode.scrollTop = chatFeed.parentNode.scrollHeight;

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: messageText })
            });

            if (!response.ok) {
                throw new Error("API call failed.");
            }

            const data = await response.json();
            
            // Hide thinking animation & reset Ollama status in Navbar
            chatLoading.style.display = "none";
            if (ollamaIcon) {
                ollamaIcon.style.color = "";
                ollamaIcon.classList.remove("pulse");
            }
            ollamaStatusLabel.textContent = "Idle";

            // Append agent response
            appendMessage("assistant", data.response);

            // Update ML diagnostics with Top 3 classification scores
            const confidencePercentage = data.all_intents ? Math.round(data.all_intents[0].confidence * 100) : 0;
            if (intentRankList && data.all_intents) {
                intentRankList.innerHTML = "";
                data.all_intents.forEach(item => {
                    const pct = Math.round(item.confidence * 100);
                    const rankItem = document.createElement("div");
                    rankItem.className = "intent-rank-item";
                    rankItem.innerHTML = `
                        <div class="intent-rank-meta">
                            <span class="intent-rank-name">${item.intent}</span>
                            <span class="intent-rank-score">${pct}%</span>
                        </div>
                        <div class="progress-track">
                            <div class="progress-fill" style="width: ${pct}%"></div>
                        </div>
                    `;
                    intentRankList.appendChild(rankItem);
                });
            }
            
            document.querySelector("#cardML .empty-state").classList.add("hidden");
            document.querySelector("#cardML .telemetry-data").classList.remove("hidden");

            // Update LLM tool diagnostics
            const hasToolCall = data.tool_calls && data.tool_calls.length > 0;
            lblParamsTable.innerHTML = ""; // Clear parameters

            // Update Pipeline Graph Nodes Dynamically
            if (graphUserQuery) graphUserQuery.textContent = `"${data.message}"`;
            if (graphMLIntent) graphMLIntent.textContent = `${data.classified_intent} (${confidencePercentage}%)`;
            if (graphPromptContext) graphPromptContext.textContent = `System Prompt + Hint: "${data.classified_intent}"`;

            if (hasToolCall) {
                const call = data.tool_calls[0];
                lblAction.textContent = `${call.tool}()`;
                lblToolOutput.textContent = call.result;

                if (graphAgentAction) graphAgentAction.textContent = `Invokes ${call.tool}()`;
                if (graphDBUpdate) graphDBUpdate.textContent = `State mutated (${call.tool})`;

                // Render parameters inside table
                if (call.args && Object.keys(call.args).length > 0) {
                    for (const [key, value] of Object.entries(call.args)) {
                        const tr = document.createElement("tr");
                        tr.innerHTML = `<td>${key}</td><td>${value}</td>`;
                        lblParamsTable.appendChild(tr);
                    }
                } else {
                    lblParamsTable.innerHTML = "<tr><td colspan='2' class='italic'>No arguments extracted</td></tr>";
                }

                // Sync simulated UI database state based on what tool was called
                if (call.tool === "freeze_card") {
                    await refreshDBState();
                } else if (call.tool === "check_balance") {
                    await refreshDBState("balance");
                } else if (call.tool === "transaction_history") {
                    await refreshDBState("transactions");
                }

                // Auto-switch to "ledger" tab to show database update visually!
                switchTab("ledger");

            } else {
                lblAction.textContent = "Direct Response (No Tool)";
                lblParamsTable.innerHTML = "<tr><td colspan='2' class='italic'>N/A (policy lookup)</td></tr>";
                lblToolOutput.textContent = "Policy query resolved directly by prompt constraints";

                if (graphAgentAction) graphAgentAction.textContent = "Direct response generated";
                if (graphDBUpdate) graphDBUpdate.textContent = "No state modifications";

                // Auto-switch to "telemetry" tab to show model reasoning stats!
                switchTab("telemetry");
            }
            
            document.querySelector("#cardAgent .empty-state").classList.add("hidden");
            document.querySelector("#cardAgent .telemetry-data").classList.remove("hidden");

        } catch (error) {
            console.error(error);
            chatLoading.style.display = "none";
            if (ollamaIcon) {
                ollamaIcon.style.color = "var(--color-red)";
                ollamaIcon.classList.remove("pulse");
            }
            ollamaStatusLabel.textContent = "Error";
            appendMessage("assistant", "Sorry, I ran into an error communicating with the agent server. Please make sure the FastAPI backend is running.");
        }
    });
});
