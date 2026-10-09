/**
 * Premier League Predictor — Minimalist Frontend Application Logic
 * Integrates with AIMLite Zero-Path Inference Engine & Open Source Crests
 */

const CLUB_LOGOS = {
  "Arsenal": "https://crests.football-data.org/57.png",
  "Aston Villa": "https://crests.football-data.org/58.png",
  "Chelsea": "https://crests.football-data.org/61.png",
  "Everton": "https://crests.football-data.org/62.png",
  "Fulham": "https://crests.football-data.org/63.png",
  "Liverpool": "https://crests.football-data.org/64.png",
  "Manchester City": "https://crests.football-data.org/65.png",
  "Manchester United": "https://crests.football-data.org/66.png",
  "Newcastle United": "https://crests.football-data.org/67.png",
  "Tottenham Hotspur": "https://crests.football-data.org/73.png",
  "Wolverhampton Wanderers": "https://crests.football-data.org/76.png",
  "Burnley": "https://crests.football-data.org/328.png",
  "Leicester City": "https://crests.football-data.org/338.png",
  "Southampton": "https://crests.football-data.org/340.png",
  "Leeds United": "https://crests.football-data.org/341.png",
  "Watford": "https://crests.football-data.org/346.png",
  "Crystal Palace": "https://crests.football-data.org/354.png",
  "Sheffield United": "https://crests.football-data.org/356.png",
  "Brighton & Hove Albion": "https://crests.football-data.org/397.png",
  "Brentford": "https://crests.football-data.org/402.png",
  "West Ham United": "https://crests.football-data.org/563.png",
  "AFC Bournemouth": "https://crests.football-data.org/1044.png",
  "Nottingham Forest": "https://crests.football-data.org/351.png",
  "Luton Town": "https://crests.football-data.org/389.png",
  "Ipswich Town": "https://crests.football-data.org/349.png",
};

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const yearInput = document.getElementById("year-input");
  const yearSeasonHint = document.getElementById("year-season-hint");
  const predictBtn = document.getElementById("predict-btn");
  const btnSpinner = document.getElementById("btn-spinner");
  const standingsTbody = document.getElementById("standings-tbody");
  const teamSearch = document.getElementById("team-search");
  const filterBtns = document.querySelectorAll(".filter-btn");
  const infoModalTrigger = document.getElementById("info-modal-trigger");
  const infoModal = document.getElementById("info-modal");
  const modalClose = document.getElementById("modal-close");
  const teamDrawer = document.getElementById("team-drawer");
  const drawerContent = document.getElementById("drawer-content");

  // Winner Elements
  const winnerSeasonLabel = document.getElementById("winner-season-label");
  const winnerLogo = document.getElementById("winner-logo");
  const winnerName = document.getElementById("winner-name");
  const winnerStadium = document.getElementById("winner-stadium");
  const winnerManager = document.getElementById("winner-manager");
  const winnerMarginText = document.getElementById("winner-margin-text");
  const winnerPoints = document.getElementById("winner-points");
  const winnerProb = document.getElementById("winner-prob");
  const winnerRecord = document.getElementById("winner-record");
  const winnerGd = document.getElementById("winner-gd");
  const winnerGoals = document.getElementById("winner-goals");

  // KPI Elements
  const kpiUcl = document.getElementById("kpi-ucl");
  const kpiAttack = document.getElementById("kpi-attack");
  const kpiAttackGoals = document.getElementById("kpi-attack-goals");
  const kpiDefense = document.getElementById("kpi-defense");
  const kpiDefenseGoals = document.getElementById("kpi-defense-goals");
  const kpiRelegation = document.getElementById("kpi-relegation");
  const tableSeasonTag = document.getElementById("table-season-tag");

  let currentStandings = [];
  let activeFilter = "all";
  let activeSearch = "";

  function getSeasonLabel(year) {
    const y = parseInt(year, 10);
    if (isNaN(y)) return "Season";
    return `${y - 1}/${String(y).slice(-2)}`;
  }

  function getClubLogo(teamName, fallbackUrl) {
    return fallbackUrl || CLUB_LOGOS[teamName] || "https://crests.football-data.org/PL.png";
  }

  // Update season label as user types (clamped 2019 to 2029)
  yearInput.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    if (!isNaN(val) && val >= 2019 && val <= 2029) {
      yearSeasonHint.textContent = `${getSeasonLabel(val)} Season • Ready to Forecast`;
    } else {
      yearSeasonHint.textContent = `Valid Season Range: 2019 – 2029`;
    }
  });

  yearInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      triggerPrediction();
    }
  });

  predictBtn.addEventListener("click", () => {
    triggerPrediction();
  });

  // Trigger Prediction (limited strictly up to 2029)
  async function triggerPrediction() {
    let yearVal = parseInt(yearInput.value, 10);
    if (isNaN(yearVal)) {
      yearVal = 2026;
      yearInput.value = 2026;
    } else if (yearVal < 2019) {
      yearVal = 2019;
      yearInput.value = 2019;
    } else if (yearVal > 2029) {
      yearVal = 2029;
      yearInput.value = 2029;
    }

    yearSeasonHint.textContent = `${getSeasonLabel(yearVal)} Season • Forecast Active`;
    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({
          year: yearVal,
          simulations: 500,
        }),
      });

      if (!response.ok) {
        throw new Error(`Inference server responded with ${response.status}`);
      }

      const rawData = await response.json();
      const payload = rawData.result || rawData;

      renderPrediction(payload);
    } catch (err) {
      console.error("Prediction Error:", err);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      btnSpinner.classList.remove("hidden");
      predictBtn.disabled = true;
      predictBtn.style.opacity = "0.7";
    } else {
      btnSpinner.classList.add("hidden");
      predictBtn.disabled = false;
      predictBtn.style.opacity = "1";
    }
  }

  // Render Prediction Result
  function renderPrediction(data) {
    if (!data || !data.standings || !data.winner) return;

    const winner = data.winner;
    const standings = data.standings;
    const summary = data.summary || {};
    currentStandings = standings;

    // 1. Winner Card
    winnerSeasonLabel.textContent = `${data.season || getSeasonLabel(data.year)} CHAMPION`;
    winnerName.textContent = winner.team;
    winnerStadium.textContent = winner.stadium || `${winner.team} Ground`;
    winnerManager.textContent = winner.manager || "Head Coach";

    winnerLogo.src = getClubLogo(winner.team, winner.logo_url);
    winnerLogo.alt = `${winner.team} Crest`;

    const marginPts = winner.title_margin_pts !== undefined 
      ? winner.title_margin_pts 
      : (winner.points - (standings[1]?.points || winner.points - 2));
    const runnerUpName = winner.runner_up || standings[1]?.team || "Second Place";
    winnerMarginText.textContent = marginPts > 0 
      ? `+${marginPts} PTS clear of runner-up ${runnerUpName}`
      : `Decided on goal difference over ${runnerUpName}`;

    winnerPoints.textContent = winner.points;
    winnerProb.textContent = winner.title_probability || "52.0%";
    winnerRecord.textContent = `${winner.won} - ${winner.drawn} - ${winner.lost}`;
    winnerGd.textContent = (winner.goal_difference > 0 ? `+${winner.goal_difference}` : winner.goal_difference);
    winnerGoals.textContent = `${winner.goals_for} GF / ${winner.goals_against} GA`;

    // 2. Key KPI Section
    const top4 = standings.slice(0, 4).map(s => s.short || s.team.slice(0, 3).toUpperCase());
    kpiUcl.textContent = top4.join(", ");

    const bestAttack = summary.top_attack || standings.reduce((prev, cur) => (cur.gf > prev.gf ? cur : prev)).team;
    const attackTeam = standings.find(s => s.team === bestAttack) || standings[0];
    kpiAttack.textContent = attackTeam.team;
    kpiAttackGoals.textContent = `${attackTeam.gf} Goals Scored`;

    const bestDef = summary.best_defense || standings.reduce((prev, cur) => (cur.ga < prev.ga ? cur : prev)).team;
    const defTeam = standings.find(s => s.team === bestDef) || standings[1] || standings[0];
    kpiDefense.textContent = defTeam.team;
    kpiDefenseGoals.textContent = `${defTeam.ga} Goals Conceded`;

    const relegated = standings.slice(17).map(s => s.short || s.team.slice(0, 3).toUpperCase());
    kpiRelegation.textContent = relegated.join(", ");

    tableSeasonTag.textContent = `${data.season || getSeasonLabel(data.year)} Season`;

    // 3. Render Table
    renderTable();
  }

  // Render Table
  function renderTable() {
    standingsTbody.innerHTML = "";

    const query = activeSearch.toLowerCase().trim();

    const filtered = currentStandings.filter((row) => {
      if (activeFilter === "ucl" && row.rank > 4) return false;
      if (activeFilter === "uel" && (row.rank < 5 || row.rank > 6)) return false;
      if (activeFilter === "relegation" && row.rank < 18) return false;

      if (query) {
        const matchesName = row.team.toLowerCase().includes(query);
        const matchesShort = (row.short || "").toLowerCase().includes(query);
        if (!matchesName && !matchesShort) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="14" style="text-align: center; padding: 24px; color: var(--text-tertiary);">No clubs found matching "${query}".</td>`;
      standingsTbody.appendChild(tr);
      return;
    }

    filtered.forEach((row) => {
      const tr = document.createElement("tr");
      
      let rankRowClass = "row-rank";
      if (row.rank <= 4) rankRowClass += " rank-ucl";
      else if (row.rank === 5) rankRowClass += " rank-uel";
      else if (row.rank === 6) rankRowClass += " rank-uecl";
      else if (row.rank >= 18) rankRowClass += " rank-relegation";
      tr.className = rankRowClass;

      tr.addEventListener("click", () => openTeamDrawer(row));

      const gdClass = row.gd > 0 ? "td-gd pos" : (row.gd < 0 ? "td-gd neg" : "td-gd");
      const gdSign = row.gd > 0 ? `+${row.gd}` : row.gd;

      const formHtml = (row.form || ["W", "D", "W", "L", "W"])
        .slice(0, 5)
        .map((r) => {
          const c = r === "W" ? "form-w" : (r === "D" ? "form-d" : "form-l");
          return `<span class="form-pill ${c}">${r}</span>`;
        })
        .join("");

      const titleTag = row.title_prob > 0.1 ? `${row.title_prob}%` : `—`;
      const uclTag = row.ucl_prob > 0.5 ? `${row.ucl_prob}%` : `—`;
      const relTag = row.relegation_prob > 0.5 ? `${row.relegation_prob}%` : `—`;

      const logoUrl = getClubLogo(row.team, row.logo_url);

      tr.innerHTML = `
        <td class="td-rank">
          <span class="rank-text">${row.rank}</span>
        </td>
        <td class="td-club">
          <div class="club-cell">
            <img 
              src="${logoUrl}" 
              alt="${row.team}" 
              class="club-crest-img"
              loading="lazy"
              onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
            />
            <div class="club-fallback-crest" style="display: none;">
              ${row.short || row.team.slice(0, 3)}
            </div>
            <div class="club-meta">
              <span class="club-name">${row.team}</span>
              <span class="club-stadium">${row.stadium || 'Premier League Ground'}</span>
            </div>
          </div>
        </td>
        <td class="td-num">${row.played || 38}</td>
        <td class="td-num">${row.won}</td>
        <td class="td-num">${row.drawn}</td>
        <td class="td-num">${row.lost}</td>
        <td class="td-num">${row.gf}</td>
        <td class="td-num">${row.ga}</td>
        <td class="${gdClass} td-num">${gdSign}</td>
        <td class="td-pts">${row.points}</td>
        <td class="td-form">
          <div class="form-streak">${formHtml}</div>
        </td>
        <td class="td-prob">${titleTag}</td>
        <td class="td-prob">${uclTag}</td>
        <td class="td-prob">${relTag}</td>
      `;

      standingsTbody.appendChild(tr);
    });
  }

  // Filter buttons
  filterBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      filterBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeFilter = btn.getAttribute("data-filter");
      renderTable();
    });
  });

  // Search input
  teamSearch.addEventListener("input", (e) => {
    activeSearch = e.target.value;
    renderTable();
  });

  // Minimal team detail drawer
  function openTeamDrawer(team) {
    const gdSign = team.gd > 0 ? `+${team.gd}` : team.gd;
    const logoUrl = getClubLogo(team.team, team.logo_url);

    drawerContent.innerHTML = `
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 14px;">
          <img src="${logoUrl}" alt="${team.team}" style="width: 38px; height: 38px; object-fit: contain;" />
          <div>
            <h3 style="font-family: var(--font-heading); font-size: 1.25rem; color: var(--text-primary); font-weight: 700;">${team.team}</h3>
            <span style="font-size: 0.78rem; color: var(--text-tertiary);">${team.stadium || 'Ground'}</span>
          </div>
        </div>
        <button id="drawer-close" style="background: transparent; border: none; font-size: 1.4rem; color: var(--text-tertiary); cursor: pointer;">&times;</button>
      </div>

      <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 16px;">
        <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
          <span style="font-size: 0.7rem; color: var(--text-tertiary); text-transform: uppercase;">Position</span>
          <div style="font-family: var(--font-heading); font-size: 1.3rem; font-weight: 800; color: var(--text-primary);">#${team.rank}</div>
          <span style="font-size: 0.72rem; color: var(--text-secondary);">${team.zone_label || 'Premier League'}</span>
        </div>
        <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
          <span style="font-size: 0.7rem; color: var(--text-tertiary); text-transform: uppercase;">Points</span>
          <div style="font-family: var(--font-heading); font-size: 1.3rem; font-weight: 800; color: var(--text-primary);">${team.points} PTS</div>
          <span style="font-size: 0.72rem; color: var(--text-tertiary);">GD: ${gdSign}</span>
        </div>
      </div>

      <div style="background: var(--bg-card-subtle); padding: 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span style="color: var(--text-tertiary);">Manager</span>
          <strong style="color: var(--text-primary);">${team.manager || 'Head Coach'}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span style="color: var(--text-tertiary);">Record</span>
          <strong style="color: var(--text-primary);">${team.won}W - ${team.drawn}D - ${team.lost}L</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span style="color: var(--text-tertiary);">Title Probability</span>
          <strong style="color: var(--text-primary);">${team.title_prob}%</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span style="color: var(--text-tertiary);">Top 4 Probability</span>
          <strong style="color: var(--text-primary);">${team.ucl_prob}%</strong>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
          <span style="color: var(--text-tertiary);">Relegation Risk</span>
          <strong style="color: var(--text-primary);">${team.relegation_prob}%</strong>
        </div>
      </div>
    `;

    teamDrawer.classList.remove("hidden");
    document.getElementById("drawer-close").addEventListener("click", () => {
      teamDrawer.classList.add("hidden");
    });
  }

  teamDrawer.addEventListener("click", (e) => {
    if (e.target === teamDrawer) {
      teamDrawer.classList.add("hidden");
    }
  });

  // Modal
  infoModalTrigger.addEventListener("click", () => {
    infoModal.classList.remove("hidden");
  });

  modalClose.addEventListener("click", () => {
    infoModal.classList.add("hidden");
  });

  infoModal.addEventListener("click", (e) => {
    if (e.target === infoModal) {
      infoModal.classList.add("hidden");
    }
  });

  // Initial load
  triggerPrediction();
});
