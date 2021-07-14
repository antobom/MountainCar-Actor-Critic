agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1)):
    
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False

    while not done:
        if np.random.random()>epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE)
        new_state, reward, done = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGAE_STATE_EVERY == 0:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state

        step +=1
        

