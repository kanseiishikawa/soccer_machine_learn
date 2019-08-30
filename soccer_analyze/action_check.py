# -*- coding: utf-8 -*-
#from world_model import World as wm
#from world_model import Vector2D
#from world_model import Line2D
#from world_model import Rect2D
#from world_model import Circle2D
import world_model
import screenshot

#ドリブルが１ パスが２ シュートが３

if __name__ == '__main__':
    wm = world_model.World( "file_name" )
    action_cycle=[]
    action=[]
    left_score = 0
    right_score = 0
    action_storage = []
    while wm.time().kick_off() <= wm.time().cycle() \
          and wm.time().cycle() <= wm.time().time_over() :
          #print(wm.time().cycle())
          finish = 0
          goal = 0

          if wm.gameMode().type() == "goal_l"\
             and wm.time().cycle() > 30:#ゴールが入ったら
              for cycle in range(1,15):
                  for unum in range(2,12):
                      if (wm.theirPlayer(unum, wm.time().cycle() - cycle).Action()== "kick"\
                         or wm.ourPlayer(unum, wm.time().cycle() - cycle).Action()== "kick")\
                         and goal == 0:
                          action_cycle.append(wm.time().cycle())
                          action.append(3) #シュー卜
                          right_score = wm.gameMode().scoreRight()
                          goal = 1
                          break

          for mate_unum in range(2,12):
              if wm.ourPlayer(mate_unum).Action() == "kick" \
                 and wm.time().cycle() < 6000 \
                 and wm.time().cycle() > 30\
                 and wm.gameMode().type() == "play_on":
                 kick_unum = mate_unum
                 if finish == 1:
                     break
                 for cycle in range(wm.time().cycle()+1,wm.time().cycle()+26):
                     if finish == 1\
                         or cycle == 3000\
                         or cycle == 6000:
                         break
                     for unum in range(2,12):
                         if wm.ourPlayer(unum , cycle).Action() == "kick"\
                            and unum == mate_unum:
                            action_cycle.append(wm.time().cycle())
                            action.append(1) #ドリブル
                            finish = 1
                            add_cycle = cycle - wm.time().cycle()
                            break

                         elif wm.ourPlayer(unum , cycle).Action() == "kick"\
                              and not unum == mate_unum:
                              action_cycle.append(wm.time().cycle())
                              action.append(2) #パス
                              finish = 1
                              add_cycle = cycle - wm.time().cycle()
                              break

                         elif wm.theirPlayer(unum , cycle).Action() == "kick"\
                              and wm.ourPlayer(kick_unum , cycle).distFromBall() > 1.3:
                              action_cycle.append(wm.time().cycle())
                              action.append(2) #パス失敗
                              finish = 1
                              add_cycle = cycle - wm.time().cycle()
                              break

                         elif wm.theirPlayer(unum , cycle).Action() == "kick"\
                              and wm.ourPlayer(kick_unum , cycle).distFromBall() < 1.3:
                              action_cycle.append(wm.time().cycle())
                              action.append(1) #ドリブル失敗
                              finish = 1
                              add_cycle = cycle - wm.time().cycle()
                              break




          for mate_unum in range(2,12):
              if wm.theirPlayer(mate_unum).Action() == "kick" \
                 and wm.time().cycle() < 5950 \
                 and wm.time().cycle() > 30\
                 and wm.gameMode().type() == "play_on"\
                 and finish == 0:
                 if finish == 1:
                     break
                 for cycle in range(wm.time().cycle()+1,wm.time().cycle()+26):
                     if finish == 1\
                        or cycle == 3000\
                        or cycle == 6000:
                           break
                     for unum in range(2,12):
                         if wm.theirPlayer(unum , cycle).Action() == "kick"\
                            and unum == mate_unum:
                            action_cycle.append(wm.time().cycle())
                            action.append(1) #ドリブル
                            finish = 1
                            add_cycle = cycle - wm.time().cycle()
                            break
                         elif wm.theirPlayer(unum , cycle).Action() == "kick"\
                              and not unum == mate_unum:
                              action_cycle.append(wm.time().cycle())
                              action.append(2) #パス
                              finish = 1
                              add_cycle = cycle - wm.time().cycle()
                              break
                    # プレーモードを更新
          wm.gameMode().UpdatePlayMode()
                    # サイクルを1進める
          wm.time().addTime()
          if goal == 1:
              for i in range(10):
                  wm.gameMode().UpdatePlayMode()
                  wm.time().addTime()

          if finish == 1:
              for i in range(1,add_cycle+1):
                  wm.gameMode().UpdatePlayMode()
                  wm.time().addTime()

    #action_cycle = [31, 421, 91]
    action_storage.append(action_cycle)
    action_storage.append(action)
    '''
    if action_storage[0,0] == 0:
        action_storage = np.delete(action_storage, 0, 1)
        action_storage = np.delete(action_storage, 0, 1)
    '''

    screenshot.capture(action_storage)
