import os
import pygame
import time

class Interface:
    """ Pygame interface for training TAMER """

    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
        os.environ["SDL_VIDEO_CENTERED"] = "0"

        self.screen = pygame.display.set_mode((200, 100))
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

    # def get_scalar_feedback(self):
    #     """
    #     Get human input. 'W' key for positive, 'A' key for negative.
    #     Returns: scalar reward (1 for positive, -1 for negative)
    #     """
    #     reward = 0
    #     area = None
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_w:
    #                 area = self.screen.fill((0, 255, 0))
    #                 reward = 1
    #                 break
    #             elif event.key == pygame.K_a:
    #                 area = self.screen.fill((255, 0, 0))
    #                 reward = -1
    #                 break
    #     pygame.display.update(area)
    #     return reward

    def get_scalar_feedback(self):
        """
        Get human input. Left button for positive feedback, right button for negative feedback.
        Return value:scalar reward (1 for positive feedback, -1 for negative feedback)
        """
        reward = 0
        area = None
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    area = self.screen.fill((0, 255, 0)) 
                    reward = 1
                    break
                elif event.button == 3:  # 右键
                    area = self.screen.fill((255, 0, 0))  
                    reward = -1
                    break
        pygame.display.update(area)
        return reward


    def show_action(self, action):
        """
        Display the agent's actions on the pygame interface
        parameters on the pygame interface:
            action: action in numeric form (currently only used in MountainCar environments)
        """
        # 清空屏幕并更新背景
        area = self.screen.fill((0, 0, 0))
        pygame.display.update(area)

        # 显示动作文本
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        area = self.screen.blit(text, text_rect)
        pygame.display.update(area)

        # # # 保持显示 0.5 秒（可以根据需要调整）
        # time.sleep(0.3)
