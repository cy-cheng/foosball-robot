# Alternative Training Algorithms and Approaches

## Overview

This document outlines alternative RL algorithms and training approaches that could potentially improve upon the current PPO-based training system. Each approach is evaluated for suitability to the foosball task.

---

## 🎯 Current Approach: PPO

**Strengths**:
- ✅ Stable and reliable
- ✅ Sample efficient for on-policy methods
- ✅ Works well with continuous control
- ✅ Handles high-dimensional observations

**Weaknesses**:
- ❌ On-policy only (can't reuse old data)
- ❌ Requires many environment interactions
- ❌ May struggle with sparse rewards (before our fixes)

**Current Performance** (after fixes):
- Expected to achieve 15-25 goals per side in Stage 4
- Training time: 3-5 hours for full curriculum
- Sample efficiency: Moderate

---

## 🚀 RECOMMENDED ALTERNATIVES

### 1. SAC (Soft Actor-Critic) ⭐⭐⭐⭐⭐

**Best alternative for this task**

**Algorithm**: Off-policy actor-critic with maximum entropy objective

**Why it's better for foosball**:
1. ✅ **Off-policy**: Can reuse experience from replay buffer (3-4× more sample efficient)
2. ✅ **Maximum entropy**: Encourages exploration naturally
3. ✅ **Continuous control**: Designed specifically for continuous actions
4. ✅ **Stable learning**: Automatic temperature tuning
5. ✅ **Better sample efficiency**: Learns faster from same amount of data

**Implementation**:
```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,      # Replay buffer
    learning_starts=1000,      # Start learning after 1K steps
    batch_size=256,
    tau=0.005,                 # Soft update coefficient
    gamma=0.99,
    train_freq=1,              # Update every step
    gradient_steps=1,
    ent_coef='auto',           # Automatic temperature tuning
    policy_kwargs=dict(
        net_arch=[256, 256],   # Larger network
    ),
    verbose=1,
)
```

**Expected improvements**:
- 2-3× faster learning (same performance in 100K vs 300K steps)
- More exploration → discovers creative strategies
- Better final performance: 20-35 goals per side

**Downsides**:
- Slightly more memory usage (replay buffer)
- Harder to debug (more hyperparameters)

**Recommendation**: ⭐⭐⭐⭐⭐ **STRONGLY RECOMMENDED** for Stage 1-3

---

### 2. TD3 (Twin Delayed DDPG) ⭐⭐⭐⭐

**Second best alternative**

**Algorithm**: Off-policy actor-critic with twin Q-networks and delayed policy updates

**Why it's good for foosball**:
1. ✅ **Off-policy**: Replay buffer for sample efficiency
2. ✅ **Reduced overestimation**: Twin Q-networks
3. ✅ **Continuous actions**: Designed for continuous control
4. ✅ **Deterministic policy**: More stable than stochastic
5. ✅ **Simple and robust**: Fewer hyperparameters than SAC

**Implementation**:
```python
from stable_baselines3 import TD3

model = TD3(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    action_noise=None,  # Or add noise: NormalActionNoise
    policy_kwargs=dict(
        net_arch=[256, 256]
    ),
    verbose=1,
)
```

**Expected improvements**:
- 2× faster learning than PPO
- Very stable training
- Good final performance: 18-30 goals per side

**Downsides**:
- Deterministic policy may explore less
- Requires careful noise tuning

**Recommendation**: ⭐⭐⭐⭐ **RECOMMENDED** as SAC alternative

---

### 3. Curriculum Learning Enhancement ⭐⭐⭐⭐⭐

**Enhancement to any algorithm**

**Approach**: More granular curriculum with automatic progression

**Current curriculum**:
1. Stage 1: Stationary ball
2. Stage 2: Rolling ball
3. Stage 3: Fast shot
4. Stage 4: Full game

**Improved curriculum**:
```
Stage 1a: Stationary ball, close range (10K steps)
Stage 1b: Stationary ball, medium range (10K steps)
Stage 1c: Stationary ball, full range (30K steps)

Stage 2a: Slow rolling ball (20K steps)
Stage 2b: Medium speed ball (20K steps)
Stage 2c: Fast rolling ball (20K steps)

Stage 3a: Medium shot speed (20K steps)
Stage 3b: Fast shot speed (30K steps)

Stage 4a: Self-play with frozen opponent (50K steps)
Stage 4b: Self-play with updating opponent (remaining)
```

**Implementation**:
```python
def get_curriculum_params(stage, substage):
    params = {
        '1a': {'ball_range': 0.2, 'ball_vel': 0},
        '1b': {'ball_range': 0.4, 'ball_vel': 0},
        '1c': {'ball_range': 0.6, 'ball_vel': 0},
        '2a': {'ball_vel_range': (0.5, 1.0)},
        '2b': {'ball_vel_range': (1.0, 2.0)},
        '2c': {'ball_vel_range': (2.0, 3.0)},
        '3a': {'shot_speed': (2.0, 3.0)},
        '3b': {'shot_speed': (3.0, 4.5)},
        '4a': {'self_play': True, 'freeze_opponent': True},
        '4b': {'self_play': True, 'freeze_opponent': False},
    }
    return params[f'{stage}{substage}']

# Auto-advance based on performance
def should_advance(mean_reward, goals_per_episode):
    return goals_per_episode >= 10 or mean_reward >= threshold
```

**Expected improvements**:
- Smoother learning curve
- Faster overall training
- Better generalization
- Fewer training failures

**Recommendation**: ⭐⭐⭐⭐⭐ **HIGHLY RECOMMENDED** to combine with any algorithm

---

### 4. Reward Shaping with HER (Hindsight Experience Replay) ⭐⭐⭐⭐

**For sparse reward environments**

**Approach**: Learn from "failures" by relabeling goals

**Concept**: When agent fails to score, pretend it was trying to reach wherever the ball ended up

**Implementation**:
```python
from stable_baselines3 import HerReplayBuffer, SAC

model = SAC(
    "MultiInputPolicy",  # Needed for HER
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    ),
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    verbose=1,
)
```

**Expected improvements**:
- Learn from every episode (even goalless ones)
- Much faster learning in early stages
- Better exploration

**Downsides**:
- Requires reformulating env as goal-conditioned
- More complex implementation

**Recommendation**: ⭐⭐⭐⭐ **RECOMMENDED** for sparse reward scenarios

---

### 5. DreamerV3 (Model-Based RL) ⭐⭐⭐

**Most sample efficient, but complex**

**Algorithm**: Learn world model, plan in imagination

**Why it could work**:
1. ✅ **Sample efficiency**: 10-100× more efficient than model-free
2. ✅ **Physics**: Foosball has clear physics that can be learned
3. ✅ **Planning**: Can plan multi-step strategies

**Expected improvements**:
- Could achieve same performance in 10K-50K steps instead of 1M
- Better long-term strategy
- More human-like play

**Downsides**:
- ❌ Very complex to implement
- ❌ Requires careful tuning
- ❌ May not be worth the effort for this task

**Recommendation**: ⭐⭐⭐ **INTERESTING** but probably overkill

---

## 📊 ALGORITHM COMPARISON

| Algorithm | Sample Efficiency | Stability | Ease of Use | Best For | Rating |
|-----------|------------------|-----------|-------------|----------|--------|
| **PPO** (current) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | General purpose | ⭐⭐⭐⭐ |
| **SAC** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Continuous control | ⭐⭐⭐⭐⭐ |
| **TD3** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Continuous control | ⭐⭐⭐⭐ |
| **HER+SAC** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Sparse rewards | ⭐⭐⭐⭐ |
| **DreamerV3** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Max efficiency | ⭐⭐⭐ |

---

## 🎓 HYBRID APPROACHES

### Approach 1: PPO → SAC Transition
**Best of both worlds**

**Strategy**:
1. Stage 1-2: Use PPO (our fixed version) for stable initial learning
2. Stage 3-4: Switch to SAC for better sample efficiency and exploration

**Implementation**:
```python
# Stage 1-2: PPO
ppo_model = PPO("MlpPolicy", env, **ppo_kwargs)
ppo_model.learn(total_timesteps=500_000)

# Extract policy
policy_state = ppo_model.policy.state_dict()

# Stage 3-4: SAC (initialize from PPO policy)
sac_model = SAC("MlpPolicy", env, **sac_kwargs)
# Transfer actor weights (may need adjustment)
sac_model.policy.load_state_dict(policy_state, strict=False)
sac_model.learn(total_timesteps=500_000)
```

**Benefits**:
- Stable early training (PPO)
- Efficient late training (SAC)
- Best overall performance

---

### Approach 2: Ensemble Self-Play
**Multiple agents with different algorithms**

**Strategy**:
Train 3 agents simultaneously:
1. Agent A: PPO (stable baseline)
2. Agent B: SAC (aggressive explorer)
3. Agent C: TD3 (deterministic tactician)

During self-play, randomly select opponent from the ensemble.

**Benefits**:
- More diverse strategies
- Robust to opponent variations
- Better generalization

**Implementation**:
```python
agents = {
    'ppo': PPO("MlpPolicy", env, **ppo_kwargs),
    'sac': SAC("MlpPolicy", env, **sac_kwargs),
    'td3': TD3("MlpPolicy", env, **td3_kwargs),
}

# Train in rotation
for iteration in range(1000):
    for name, agent in agents.items():
        # Select random opponent
        opponent = random.choice([a for a in agents.values() if a != agent])
        env.set_opponent(opponent)
        agent.learn(total_timesteps=10_000)
```

---

## 🔧 IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (1-2 days)
1. ✅ **Enhanced Curriculum** (already proposed above)
   - Add substages to existing curriculum
   - Implement automatic advancement
   - Expected: 30% faster training

2. ✅ **Reward Component Logging**
   - Add individual reward component tracking
   - Visualize in TensorBoard
   - Debug what agent is optimizing

### Phase 2: Algorithm Upgrade (3-5 days)
3. ✅ **Switch to SAC**
   - Replace PPO with SAC for Stage 1-3
   - Expected: 2-3× faster training
   - Better exploration and sample efficiency

4. ✅ **Improved Network Architecture**
   - Larger networks [256, 256] or [128, 128, 128]
   - Add layer normalization
   - Consider attention mechanism for ball-player relations

### Phase 3: Advanced (1-2 weeks)
5. ✅ **HER Integration**
   - Reformulate as goal-conditioned task
   - Implement HER replay buffer
   - Expected: Learn from all experiences

6. ✅ **Ensemble Self-Play**
   - Train multiple agents with different algorithms
   - Use ensemble as opponent pool
   - Expected: More robust strategies

---

## 💡 SPECIFIC RECOMMENDATIONS

### For Fastest Initial Training
**Use SAC with enhanced curriculum:**
```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    ent_coef='auto',
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)

# Train with fine-grained curriculum
for substage in ['1a', '1b', '1c']:
    env.set_curriculum(substage)
    model.learn(total_timesteps=substage_steps[substage])
```

**Expected**: First goal in 10K steps, good performance by 100K steps

---

### For Best Final Performance
**Use ensemble self-play:**
```python
# Train 3 agents with different algorithms
agents = train_ensemble(['PPO', 'SAC', 'TD3'], stage_1_3_steps=300_000)

# Stage 4: Self-play with ensemble opponents
for agent in agents:
    train_self_play(agent, opponent_pool=agents, steps=500_000)

# Select best agent
best_agent = evaluate_and_select(agents)
```

**Expected**: 25-40 goals per side in final matches

---

### For Minimum Effort
**Stick with fixed PPO, add enhanced curriculum:**
```python
# Just add more curriculum stages
ENHANCED_CURRICULUM = {
    '1a': {'ball_range': 0.2, 'steps': 10_000},
    '1b': {'ball_range': 0.4, 'steps': 10_000},
    '1c': {'ball_range': 0.6, 'steps': 30_000},
    # ... etc
}

# Auto-advance based on goals
for stage, params in ENHANCED_CURRICULUM.items():
    train_until_goals(model, env, min_goals=10, max_steps=params['steps'])
```

**Expected**: 20-30% improvement over current training

---

## 📚 RESEARCH PAPERS

### Key Papers to Read

1. **SAC**: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
   - Haarnoja et al., 2018
   - Maximum entropy RL for continuous control

2. **TD3**: [Addressing Function Approximation Error](https://arxiv.org/abs/1802.09477)
   - Fujimoto et al., 2018
   - Improved DDPG with twin Q-networks

3. **HER**: [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
   - Andrychowicz et al., 2017
   - Learning from failures in sparse reward environments

4. **DreamerV3**: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
   - Hafner et al., 2023
   - Model-based RL with world models

5. **Curriculum Learning**: [Automatic Curriculum Learning](https://arxiv.org/abs/2003.04960)
   - Portelas et al., 2020
   - Adaptive curriculum for RL

---

## 🎯 FINAL RECOMMENDATION

### For Immediate Implementation

**Option A: Conservative (Stick with PPO)**
✅ Use fixed PPO from this PR  
✅ Add enhanced curriculum  
✅ Add reward logging  
⏱️ Estimated effort: 1 day  
📈 Expected improvement: 20-30% faster training

**Option B: Moderate (Switch to SAC)** ⭐ **RECOMMENDED**
✅ Replace PPO with SAC  
✅ Use enhanced curriculum  
✅ Larger network [256, 256]  
⏱️ Estimated effort: 3 days  
📈 Expected improvement: 2-3× faster training, better final performance

**Option C: Aggressive (Full Ensemble)**
✅ Train PPO, SAC, TD3 ensemble  
✅ Enhanced curriculum  
✅ Self-play with ensemble opponents  
⏱️ Estimated effort: 1-2 weeks  
📈 Expected improvement: 3-4× faster, significantly better final performance

---

## 🔬 TESTING PROTOCOL

### For Each Algorithm

1. **Sanity Check** (10K steps):
   - Verify no crashes
   - Reward should increase
   - Agent should make contact

2. **Stage 1 Test** (100K steps):
   - Should score first goal by 30K steps
   - Mean reward > 0 by 100K steps
   - 5+ goals per episode

3. **Full Curriculum** (500K steps):
   - Complete Stage 1-3
   - Begin Stage 4
   - 10+ goals per episode

4. **Final Evaluation** (1M steps total):
   - Stage 4 matches: 15-25 goals per side
   - Win rate ~50% in symmetric matchup
   - Diverse strategies observed

---

## ✨ SUMMARY

**Quick wins** (1-2 days):
1. Enhanced curriculum → 30% improvement
2. Reward logging → Better debugging

**Medium effort** (3-5 days):
1. SAC algorithm → 2-3× improvement ⭐
2. Larger networks → 10-20% improvement

**Long term** (1-2 weeks):
1. HER integration → Better sparse reward handling
2. Ensemble self-play → Best final performance

**Recommended immediate action**: 
→ **Implement SAC with enhanced curriculum** for best ROI

Training should improve from:
- **Before fixes**: 0 goals after 1M steps
- **After PPO fixes**: 15-25 goals after 1M steps  
- **With SAC**: 20-35 goals after 300K steps ⭐

**Bottom line**: The PPO fixes are solid, but SAC would be 2-3× better. Enhanced curriculum helps any algorithm.
