import numpy as np
import copy
from tqdm import tqdm
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import SGD
from keras import metrics
from keras.layers import Dense, Flatten, Conv2D
import time

# 환경 정의
field_len = 8


class Environment():

    def __init__(self):
        # 64개의 배열 p1은 1, p2는 -1로 초기화
        self.board_a = np.zeros(field_len*field_len)
        self.turn_cnt = 0
        for i in range(field_len):
            for j in range(field_len):
                if j >= 0 and j <= (field_len/2)-1:
                    self.board_a[field_len*i+j] = 1
                else:
                    self.board_a[field_len*i+j] = -1

    # 플레이어가 선택할 수 있는 범위를 반환
    def getAction(self, player, num):
        actions = []
        for i in range(field_len*field_len):
            if (self.board_a[i] != player) and (i != num) and (i != num-field_len)\
                    and (i != num+field_len) and (i != num-1) and (i != num+1):
                actions.append(i)
        return actions

    # 게임이 종료 가능한지 확인
    def endCheck(self, player, num):
        actions = self.getAction(player*-1, num)
        standard = self.board_a[0]
        # 정해진 수를 다 둔 경우
        if self.turn_cnt >= 60:
            return True
        # 상대방이 더이상 수를 둘 수 없는 경우
        elif len(actions) == 0:
            return True
        return False

    # 점수 확인
    def checkPoint(self, player):
        win = 0
        lose = 0
        for i in range(len(self.board_a)):
            if self.board_a[i] == player:
                win += 1
            else:
                lose += 1
        #print("win : {0}, lose : {1}".format(win, lose))
        if win > lose:
            return 1
        elif win < lose:
            return -1
        else:
            return 0

    # 현재 보드의 상태를 표시 p1 = O, p2 = X

    def printBoard(self):
        print("+----+----+----+----+----+----+----+----+")
        for i in range(field_len):
            for j in range(field_len):
                if self.board_a[field_len*i+j] == 1:
                    print("| O ", end=" ")
                elif self.board_a[field_len*i+j] == -1:
                    print("| X ", end=" ")
            print("|")
            print("+----+----+----+----+----+----+----+----+")

    # 블럭 뒤집기
    def changeBlock(self, block, player):
        self.board_a[block] *= -1
        limit = field_len*field_len-1
        # 가장 왼쪽 줄에 있는 경우
        if (block % field_len) == 0:
            if (block-field_len) >= 0:
                self.board_a[block-field_len] *= -1
            if (block+field_len) <= limit:
                self.board_a[block+field_len] *= -1
            self.board_a[block+1] *= -1
        # 가장 오른쪽 줄에 있는 경우
        elif (block % field_len) == (field_len-1):
            if (block-field_len) >= 0:
                self.board_a[block-field_len] *= -1
            if (block+field_len) <= limit:
                self.board_a[block+field_len] *= -1
            self.board_a[block-1] *= -1
        else:
            if (block-field_len) >= 0:
                self.board_a[block-field_len] *= -1
            if (block+field_len) <= limit:
                self.board_a[block+field_len] *= -1
            self.board_a[block-1] *= -1
            self.board_a[block+1] *= -1

        self.turn_cnt += 1


class Random_player:
    def __init__(self):
        self.name = "Random player"

    def select_action(self, env, player, num):
        # np.random.seed(50)
        actions = env.getAction(player, num)
        action = np.random.randint(len(actions))

        return actions[action]


class DQN_player():
    def __init__(self, first_learn, name):
        self.name = "DQN_player"
        self.epsilon = 1
        self.learning_rate = 0.1
        self.gamma = 0.9

        # 신경망 생성
        if first_learn == True:
            self.main_network = self.make_network()
            self.target_network = self.make_network()
        # 신경망 모델 불러오기
        else:
            self.main_network = self.loadModel(name)
            self.target_network = self.loadModel(name)
            #
            #weights = self.main_network.get_weights()
            # print("-------불러올 때 가중치 값-------")
            # for i in weights:
            #     print(i)
            print("end laod model")

        # 가중치 복사
        self.copy_network()

    def loadModel(self, name):
        filename = name + '_main_network.h5'

        model = load_model(filename)

        return model

    def make_network(self):
        self.model = Sequential()
        # 합성곱층 생성
        self.model.add(Conv2D(8, (4, 4), padding='same',
                       activation='relu', input_shape=(field_len, field_len, 2)))
        self.model.add(Conv2D(16, (4, 4), padding='same',
                       activation='relu'))
        self.model.add(Conv2D(32, (4, 4), padding='same', activation='relu'))
        #self.model.add(Conv2D(64, (4, 4),padding='same', activation='relu'))
        # 플래튼층 생성
        self.model.add(Flatten())
        # 완전연결계층 생성
        self.model.add(Dense(512, activation='tanh'))
        self.model.add(Dense(256, activation='tanh'))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Dense(64))

        print(self.model.summary())

        self.model.compile(optimizer=SGD(lr=0.01),
                           loss='mean_squared_error', metrics=['mse'])

        return self.model

    # 신경망 복사
    def copy_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    # 신경망 저장
    def save_network(self, name):
        filename = name + '_main_network.h5'

        self.main_network.save(filename)

        # weights = self.main_network.get_weights()
        # print("-------저장할 때 가중치 값-------")
        # for i in weights:
        #     print(i)

        print("end save model")

    # 보드 상태를 2차원으로 변경
    def state_convert(self, board_a):
        d_state = np.full((field_len, field_len, 2), 0.1)
        for i in range(field_len*field_len):
            if board_a[i] == 1:
                d_state[i//field_len, i % field_len, 0] = 1
            elif board_a[i] == -1:
                d_state[i//field_len, i % field_len, 1] = 1
            else:
                pass

        return d_state

    # 행동(블록) 선택
    def select_action(self, env, player, num):
        action = self.policy(env, player, num)

        return action

    def policy(self, env, player, num):
        # 행동 가능한 상태 저장
        available_state = env.getAction(player, num)

        #
        #print("available : {}".format(available_state))

        state_2d = self.state_convert(env.board_a)
        x = np.array([state_2d], dtype=np.float32).astype(np.float32)

        # 신경망을 이용해 qvalue 계산
        qvalues = self.main_network.predict(x)[0, :]
        available_state_qvalues = qvalues[available_state]

        #
        #print("available_state_qvalues : {}".format(available_state_qvalues))

        greedy_action = np.argmax(available_state_qvalues)

        # max q값이 중복되는 경우 랜덤으로 선택
        double_check = (np.where(qvalues == np.max(
            available_state[greedy_action]), 1, 0))

        if np.sum(double_check) > 1:
            double_check = double_check/np.sum(double_check)
            greedy_action = np.random.choice(
                range(0, len(double_check)), p=double_check)

        # e-greedy
        pr = np.zeros(len(available_state))

        for i in range(len(available_state)):
            if i == greedy_action:
                pr[i] = 1 - self.epsilon + self.epsilon/len(available_state)
            else:
                pr[i] = self.epsilon / len(available_state)

        action = np.random.choice(range(0, len(available_state)), p=pr)

        return available_state[action]

    def learn_dqn(self, board_backup, action_backup, env, reward, num):
        player = 1

        # 현재 보드 상태를 2차원으로 변환 후 qvalue 계산
        new_state = self.state_convert(board_backup)
        x = np.array([new_state], dtype=np.float32).astype(np.float32)
        qvalues = self.main_network.predict(x)[0, :]

        # 현재 qvalue를 복사
        before_action_value = copy.deepcopy(qvalues)
        delta = 0

        # 게임이 종료된 경우 : 수를 다 둔 경우, p1이 더이상 둘 곳이 없는 경우
        if env.endCheck(player*-1, num) == True:
            # 신경망 학습을 위한 정답 데이터 생성
            qvalues[action_backup] = reward
            y = np.array([qvalues], dtype=np.float32).astype(np.float32)

            # 신경망 학습
            self.main_network.fit(x, y, epochs=10, verbose=0)
        # 게임이 진행중인 경우
        else:
            # 다음 상태의 max q값 계산
            new_state = self.state_convert(env.board_a)
            next_x = np.array([new_state], dtype=np.float32).astype(np.float32)
            next_qvalues = self.target_network.predict(next_x)[0, :]
            available_state = env.getAction(player, num)
            maxQ = np.max(next_qvalues[available_state])

            delta = self.learning_rate * \
                (reward + self.gamma * maxQ - qvalues[action_backup])
            qvalues[action_backup] += delta

            # 신경망 학습
            y = np.array([qvalues], dtype=np.float32).astype(np.float32)
            self.main_network.fit(x, y, epochs=10, verbose=0)


# DQN 플레이어 훈련
def training_dqn(DQN, DQN2, name):
    np.random.seed(0)

    p1_DQN = DQN
    p2 = DQN2

    p1_score = 0
    p2_score = 0
    draw_score = 0

    max_learn = 30000

    for j in tqdm(range(max_learn)):
        np.random.seed(j)
        env = Environment()
        num = -10

        p1_DQN.epsilon = 0.7
        p2.epsilon = 0
        p1_DQN.copy_network()

        for i in range(100):
            player = 1
            act = p1_DQN.policy(env, player, num)
            num = act

            p1_board_backup = tuple(env.board_a)
            p1_action_backup = act

            env.changeBlock(act, player)

            if env.endCheck(player, num) == True:
                if env.checkPoint(player) == 0:
                    p1_DQN.learn_dqn(
                        p1_board_backup, p1_action_backup, env, 0, num)
                    draw_score += 1
                    break
                elif env.checkPoint(player) == 1:
                    p1_DQN.learn_dqn(
                        p1_board_backup, p1_action_backup, env, 1, num)
                    p1_score += 1
                    break
                else:
                    p1_DQN.learn_dqn(
                        p1_board_backup, p1_action_backup, env, -1, num)
                    p2_score += 1
                    break

            player = -1
            act = p2.select_action(env, player, num)
            num = act
            env.changeBlock(act, player)

            if env.endCheck(player, num) == True:
                if env.checkPoint(player) == 0:
                    p1_DQN.learn_dqn(
                        p1_board_backup, p1_action_backup, env, 0, num)
                    draw_score += 1
                    break
                elif env.checkPoint(player) == 1:
                    p1_DQN.learn_dqn(
                        p1_board_backup, p1_action_backup, env, -1, num)
                    p2_score += 1
                    break
                else:
                    p1_DQN.learn_dqn(
                        p1_board_backup, p1_action_backup, env, 1, num)
                    p1_score += 1
                    break

            p1_DQN.learn_dqn(
                p1_board_backup, p1_action_backup, env, 0, num)

        if j % 5 == 0:
            p1_DQN.copy_network()

        if j % 5000 == 0:
            p1_DQN.save_network(name)

    print("p1={} p2={} draw={}".format(p1_score, p2_score, draw_score))

    p1_DQN.save_network(name)


first_learn = True
name = "Test_DQN_0004"

p1 = DQN_player(first_learn, name)
p2 = DQN_player(False, "Test_DQN_0002")

#p1.epsilon = 0

training_dqn(p1, p2, name)

# print(p1.main_network.get_weights())

win = 0
lose = 0
draw = 0

for k in tqdm(range(1)):
    # for k in range(100):
    cnt = 0
    # p1은 1, p2는 -1
    player = 1
    env = Environment()

    # env.printBoard()
    num = -10

    while(env.endCheck(player*-1, num) == False):

        #print("횟수:", cnt)
        #print("플레이어:", player)

        if player == 1:
            action = p1.select_action(env, player, num)
        else:
            action = p2.select_action(env, player, num)

        # if cnt < 10:
        #     print("player: {}, 선택한 좌표: {}".format(player, action))
        env.changeBlock(action, player)

        # 이전에 클릭한 좌표
        num = action

        # env.printBoard()
        cnt += 1

        player *= -1

    # env.printBoard()
    winner = env.checkPoint(1)

    # p1가 이기면 lose, p2가 이기면 win
    if winner == 1:
        lose += 1
    elif winner == -1:
        win += 1
    else:
        draw += 1

print("p1 : {0}, p2 : {1}, 무승부 : {2}".format(lose, win, draw))
