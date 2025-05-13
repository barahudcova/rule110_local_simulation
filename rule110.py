import torch
import numpy as np
import pygame
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
import re

class Rule110Universality():
    """
        Universality in rule 110.
    """

    def __init__(self, size, tape_symbols, cyclic_tag):
        """
            Parameters:
            size : 2-uple (H,W)
                Shape of the CA world
            wolfram_num : int
                Number of the wolfram rule
            random : bool
                If True, the initial tape of the automaton is random. Otherwise, the initial tape is 'YN'
        """
        self.w = size
        self.rule = self.convert_wolfram_num(110) # (8,) tensor, rule[i] is 0 if the i'th neighborhood yields 0, 1 otherwise
        self.cyclic_tag = cyclic_tag
        self.init_tape = tape_symbols

        # load dictionaries and glider patterns
        self.load_patterns()
        self.init_cyclic_tag_data()
        self.e_speed = 4/15
        self.a_speed = 2/3

        #initialize keyboard step sizes
        self.left_step_size = self.w//20
        self.right_step_size = self.w//20

        #set up world parameters
        self.offset = 100 # margin to circumvent coarse graining missing a pattern cut in half at the border
        self.world = torch.zeros((self.w+2*self.offset),dtype=torch.int) # Vector of W elements
        #self.render_size =  60 if self.h <= 600 else int(60*(self.h//600))
        self.render_size =  210
        self.worlds = torch.ones((self.render_size, self.w+2*self.offset), dtype=torch.int) # Vector of (render_size, W) elements
        self.time = 0 # Current time step, to keep track for plotting the full evolution

        #set up colors
        self.color0 = torch.tensor([1.,1.,1.])
        self.color1 = torch.tensor([0.,0.,0.])
        self.ethercolor1 = torch.tensor([0.96,0.96,0.96])
        self.ncolor0 = torch.tensor([1.,1.,1.])
        self.ncolor1 = torch.tensor([0.549, 0.071, 0.086])
        self.ycolor1 = torch.tensor([0.031, 0.322, 0.125])
        self.moving_n_color1 = torch.tensor([0.678, 0.18, 0.639])
        self.moving_y_color1 = torch.tensor([0.18, 0.267, 0.678])
        self.colors = torch.stack([self.color0, self.color1, self.ethercolor1, self.ncolor0, self.ncolor1, self.ycolor1, self.moving_n_color1, self.moving_y_color1])

        

    def load_patterns(self):
        self.dict_yn = json.load(open('utils/dict_yn.json', 'r'))
        self.dict_rl = json.load(open('utils/dict_rl.json', 'r'))
        self.dict_oss = {0: (1, 1), 1: (2, 0), 2:(0, 0)} #dictionary for ossifiers; example: if last appended ossifier is O[1], then what gets prepended is 0[2]+0*ether+(short or long distance * ether) 
        self.gliders = json.load(open('utils/gliders.json', 'r'))

        self.ether = self.gliders['ether']
        self.str_ether = self.to_str(self.ether)
        self.Y = self.gliders['Y']
        self.N = self.gliders['N']
        self.L = self.gliders['L']
        self.strL = ["".join(str(s) for s in l) for l in self.gliders['L']]
        self.C2 = self.gliders['C2']
        self.strC2 = ["".join(str(s) for s in self.ether+c+self.ether) for c in self.gliders['C2']]
        self.PC = self.gliders['PC']
        self.strPC = ["".join(str(s) for s in pc) for pc in self.gliders['PC']]
        self.SC = self.gliders['SC']
        self.strSC = ["".join(str(s) for s in sc) for sc in self.gliders['SC']]
        self.O = self.gliders['O']
        self.strO = ["".join(str(s) for s in o) for o in self.gliders['O']]
        self.RL = self.gliders['RL']
        self.strRL = ["".join(str(s) for s in rl) for rl in self.gliders['RL']]
        self.L = self.gliders['L']
        self.strE = ["".join(str(s) for s in e) if len(e)> 20 else "".join(str(s) for s in e+self.ether) for e in self.gliders['E']]
        self.strA4 = ["".join(str(s) for s in a) if len(a)> 20 else "".join(str(s) for s in a+self.ether) for a in self.gliders['A4']]
        self.Y_middle = self.gliders['Y_middle']
        self.N_middle = self.gliders['N_middle']
        self.N_outer = self.gliders['N_outer']
        self.Y_outer = self.gliders['Y_outer']

        self.N_moving_middle = self.gliders['N_moving_middle']
        self.Y_moving_middle = self.gliders['Y_moving_middle']

        self.Y_middle_torch = [torch.tensor(e, dtype=torch.uint8) for e in self.pad_patterns(self.Y_middle)]
        self.Y_outer_torch = [[torch.tensor(e1, dtype=torch.uint8), torch.tensor(e2, dtype=torch.uint8)] for e1, e2 in self.Y_outer]
        self.N_middle_torch = [torch.tensor(e, dtype=torch.uint8) for e in self.pad_patterns(self.N_middle)]
        self.N_outer_torch = [[torch.tensor(e1, dtype=torch.uint8), torch.tensor(e2, dtype=torch.uint8)] for e1, e2 in self.N_outer]
        self.N_moving_middle_torch = [torch.tensor(e, dtype=torch.uint8) for e in self.N_moving_middle]
        self.Y_moving_middle_torch = [torch.tensor(e, dtype=torch.uint8) for e in self.Y_moving_middle]
        self.ether_pattern = torch.tensor(self.ether)

        self.ether_pattern_len = len(self.ether_pattern)  
        self.y_middle_pattern_len = len(self.Y_middle_torch[0])
        self.y_outer_pattern_len = len(self.Y_outer_torch[0][0])
        self.n_middle_pattern_len = len(self.N_middle_torch[0])
        self.n_outer_pattern_len = len(self.N_outer_torch[0][0])
        self.n_moving_middle_pattern_len = len(self.N_moving_middle[0])
        self.y_moving_middle_pattern_len = len(self.Y_moving_middle[0])

        self.newC2 = [self.ether+c2+self.ether for c2 in self.C2]

    def convert_wolfram_num(self,wolfram_num : int):
        """
            Converts a wolfram number to a rule tensor.
            A tensor, with 8 elements, 0 if the rule is 0, 1 if the rule is 1.
        """
        out = torch.zeros(8,dtype=torch.int8) # Prepare my arary of 8 binary elements
        for i in range(8):
            out[i] = (wolfram_num >> i) & 1
        
        # Now the array out contains the binary representation of wolfram_num

        return out.to(dtype=torch.int) # (Array of 8 elements, where out[i]=0 if the  neighborhood number i yields 0)

    def to_str(self, config):
        if isinstance(config, np.ndarray) or isinstance(config, list):
            return "".join(str(s) for s in config)
        elif isinstance(config, torch.Tensor):
            return "".join(str(s.item()) for s in config)

    def pad_patterns(self, L):
        max_len = max([len(l) for l in L])
        for i, l in enumerate(L):
            diff = max_len-len(l)
            num_ethers = diff//len(self.ether)
            residue = diff%len(self.ether)
            L[i] = l + num_ethers*self.ether + self.ether[:residue]
        return L

    def init_cyclic_tag_data(self):
        self.max_appendant_len = max(len(a) for a in self.cyclic_tag)
        self.min_appendant_len = min(len(a) for a in self.cyclic_tag)

        self.tot_symbols = "".join(self.cyclic_tag)
        self.total_num_of_symbols = len("".join(self.cyclic_tag))

        self.num_ys = self.tot_symbols.count("Y")
        self.num_ns = self.tot_symbols.count("N")
        self.num_empty = self.cyclic_tag.count("")
        self.num_non_empty = len(self.cyclic_tag)-self.num_empty

        self.long_ossifier_distance =  int((76*self.num_ys+80*self.num_ns+60*self.num_non_empty+43*self.num_empty)//4)*4*2+3    #to be double-checked, is from Cooks paper concrete view of rule 110 computation 

    def encode_raw_leader(self, e_i):
        rl_i, num_ethers = self.dict_rl[str(e_i)]
        config=self.ether*num_ethers+self.RL[rl_i]
        return config, (rl_i+22)%30

    def encode_appendant(self, yn_seq, leader_i):
        assert (len(yn_seq)%6)==0
        config = []
        
        i=leader_i
        for char_i, C in enumerate(yn_seq):
            j, num_ether = self.dict_yn[str(i)]["0"]
            if char_i ==0:
                config+=self.ether*num_ether+self.PC[j]
                j=(j+14)%30
            else:
                config+=self.ether*num_ether+self.SC[j]
            k, num_ether = self.dict_yn[str(j)][C]
            config+=self.ether*num_ether+self.SC[k]
            i = k

        return config, i

    def encode_tape(self):
        config=3*self.ether
        for C in self.init_tape:
            if C == "Y":
                config+=self.ether+self.Y[0]+6*self.ether
            if C == "N":
                config+=self.ether+self.N[0]+7*self.ether
        leader_index=2
        config+=4*self.ether+self.L[leader_index]
        
        return config, (leader_index+18)%30

    def encode_cyclic_tag(self, num_cyclic_tag):
        config=[]
        cyclic_tag = self.cyclic_tag*num_cyclic_tag
        e_i = self.last_e_glider_index
        for appendant in cyclic_tag:
            a, e_i = self.encode_appendant(appendant, leader_i=e_i)
            config+=a+8*self.ether
            l, e_i = self.encode_raw_leader(e_i)
            config+=l
        self.last_e_glider_index = e_i
        return config

    def encode_ossifiers(self, num_ossifiers):
        oss_index, num_ether = self.dict_oss[self.oss_index]
        ossifiers = self.O[oss_index]+self.ether*(num_ether+self.long_ossifier_distance)
        #print("new oss index: ", oss_index)
        for o in range(num_ossifiers-1):
            oss_index, num_ether = self.dict_oss[oss_index]
            #print("new oss index: ", oss_index)
            ossifiers = self.O[oss_index]+self.ether*(num_ether+self.long_ossifier_distance)+ossifiers
        self.oss_index = oss_index
        return ossifiers

    def crop(self, str_c):
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while str_c.startswith(self.str_ether):
            str_c = str_c[14:]
        i = str_c.rfind(self.str_ether)
        str_c = str_c[:i]
        while str_c.endswith(self.str_ether):
            str_c = str_c[:-14]
        return [int(i) for i in str_c]

    def left_crop(self, str_c):
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while str_c.startswith(self.str_ether):
            str_c = str_c[14:]
        return [int(i) for i in str_c]

    def right_crop(self, str_c):
        i = str_c.rfind(self.str_ether)
        str_c = str_c[:i]
        while str_c.endswith(self.str_ether):
            str_c = str_c[:-14]
        return [int(i) for i in str_c]

    def left_hard_croppable(self, str_c):
        if str_c.startswith(self.str_ether):
            return True
        for e in self.strE:
            if str_c.startswith(e):
                return True

    def left_hard_crop_index(self, str_c):
        if str_c.startswith(self.str_ether):
            return len(self.str_ether)
        for e in self.strE:
            if str_c.startswith(e):
                #print("Had to crop garbage")
                return len(e)

    def starts_with_ossifier(self, str_c):
        for i, o in enumerate(self.strO):
            if str_c.startswith(o):
                self.oss_index = i
                #print("updating oss index: ", i)
                return True
        return False

    def left_crop(self, str_c):
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while str_c.startswith(self.str_ether):
            str_c = str_c[14:]
        return [int(i) for i in str_c]

    def left_hard_crop(self, str_c):
        """sometimes, cropping ether to clean up a config is not enough because garbage gliders propagate to the left and make a mess, so this funciton gets rid of them"""
        i = str_c.index(self.str_ether)
        str_c = str_c[i:]
        while self.left_hard_croppable(str_c):
            index = self.left_hard_crop_index(str_c)
            str_c = str_c[index:]

        return [int(i) for i in str_c]

    def get_init_hidden_world(self):
        #computes the closest safe distance of the first ossifier, this depends on the length of the appendants and the first occurrence of Y + some margin for each tabe symbol
        r_tape_seq = self.init_tape[::-1]
        mask = [int(i =="Y") for i in r_tape_seq]
        first_appendant_index = list(np.array(mask)*np.array([int(len(i)>0) for i in (self.cyclic_tag*len(self.init_tape))[:len(self.init_tape)]])).index(1) #finds the first instance of a non-empty appendant hitting a Y to determine the distance of the first batch of ossifiers
        first_ossifier_distance = len(self.init_tape)*28+self.max_appendant_len*150*first_appendant_index+139
        self.short_ossifier_distance = 67


        #encode first six ossifiers
        num_oss = 6
        oss_index = 2
        ossifiers = self.O[oss_index]+self.ether*first_ossifier_distance
        for _ in range(num_oss-1):
            oss_index, num_ether = self.dict_oss[oss_index]
            #print("new oss index: ", oss_index)
            ossifiers = self.O[oss_index]+self.ether*(num_ether+self.long_ossifier_distance)+ossifiers

        ossifiers = 3*self.ether+ossifiers


        #encode tape symbols        
        tape_config, e_i = self.encode_tape()
        self.last_e_glider_index = e_i
        self.oss_index = oss_index

        tape_center = len(ossifiers)+len(tape_config)
        self.left_tape_end = len(ossifiers)
        self.right_tape_end = len(ossifiers)+len(tape_config)
        
        #encode cyclic tag system
        num_cyclic_tag = int(self.w//(self.total_num_of_symbols*600))+1
        #print(f"appending {num_cyclic_tag} of tag systems")
        cyclic_tag_config = self.encode_cyclic_tag(num_cyclic_tag)
        self.len_cyclic_tag = int(len(cyclic_tag_config)//num_cyclic_tag)
        init_config = 3*self.ether+ossifiers+tape_config+cyclic_tag_config+3*self.ether


        self.hidden_world = torch.tensor(init_config, dtype=torch.int)
        self.world_center =  tape_center
        self.img_center = tape_center
        self.zero_index = tape_center
        self.action_window = (len(ossifiers), len(ossifiers)+len(tape_config))
        self.str_c = self.to_str(self.hidden_world)

    def update_hidden_world(self, prolong_right = True):
        self.str_c = self.to_str(self.hidden_world)

        #checking whether ossifiers are intact and cropping garbage gliders
        l_raw_final_config = self.left_hard_crop(self.str_c)
        if not self.starts_with_ossifier(self.to_str(l_raw_final_config)): 
            #print("skipping update ", self.time)
            return #the leftmost ossifier is just in the middle of a collision, better to skip this update and wait for the ossifier to stabilize
        

        left_offset = len(self.str_c)-len(l_raw_final_config)  
        cropped_center_index = self.world_center-left_offset

        cropped_config = self.right_crop(self.to_str(l_raw_final_config))
        str_cropped_config = self.to_str(cropped_config)

        #find the all raw leaders
        rl = []
        for s in self.strRL:
            if s in self.str_c:
                rl += [m.start() for m in re.finditer(s, self.str_c)]

        
        if len(rl) < 2 and prolong_right:
            # prolonging config to the right
            #print("updating right ", self.time)
            cyclic_tag_config = self.encode_cyclic_tag(1)
            cropped_config+=cyclic_tag_config
        

        ossifiers = []
        for o in self.strO:
            if o in str_cropped_config:
                ossifiers += [m.start() for m in re.finditer(o, str_cropped_config)]


        if len(ossifiers) < 2:
            #prolonging config to the left
            #print("updating left ", self.time)
            ossifiers = self.encode_ossifiers(2)
            cropped_config = ossifiers+cropped_config
            cropped_center_index+=len(ossifiers)

        old_center = self.world_center
        cropped_config = int(self.w//7)*self.ether+cropped_config+int(self.w//7)*self.ether

        diff = cropped_center_index+len(int(self.w//7)*self.ether)-self.world_center
        self.img_center = self.img_center+diff

        self.world_center = cropped_center_index+len(int(self.w//7)*self.ether)

        self.zero_index = cropped_center_index+len(int(self.w//7)*self.ether)
        self.hidden_world = torch.tensor(cropped_config, dtype=torch.int) 


    def coarse_grain(self, row):
        """
        Coarse-grains a single row of the input matrix.
        
        Parameters:
        row : torch.Tensor
            A single row of the input matrix (1D tensor).

        Returns:
        torch.Tensor
            The coarse-grained row.
        """
        cols = row.shape[0]
        new_row = row.clone()

        # Match ether
        windows = row.unfold(0, self.ether_pattern_len, 1)
        matches = (windows == self.ether_pattern).all(dim=1)
        for j in range(self.ether_pattern_len):
            new_row[j:cols-self.ether_pattern_len+j+1] = torch.where(
                matches & (new_row[j:cols-self.ether_pattern_len+j+1] == 1),
                2,
                new_row[j:cols-self.ether_pattern_len+j+1]
            )

        # Match outer N components
        rolled_row = torch.roll(row, -115)
        windows1 = row.unfold(0, self.n_outer_pattern_len, 1)
        windows2 = rolled_row.unfold(0, self.n_outer_pattern_len, 1)
        matches = torch.zeros(windows1.shape[0], dtype=torch.bool)
        for pattern1, pattern2 in self.N_outer_torch:
            matches |= ((windows1 == pattern1).all(dim=1) & (windows2 == pattern2).all(dim=1))
        offset = 130
        for j in range(self.n_outer_pattern_len + offset):
            new_row[j:cols-self.n_outer_pattern_len-offset+j+1] = torch.where(
                matches[:-offset] & (new_row[j:cols-self.n_outer_pattern_len-offset+j+1] <= 1),
                new_row[j:cols-self.n_outer_pattern_len-offset+j+1] + 3,
                new_row[j:cols-self.n_outer_pattern_len-offset+j+1]
            )

        # Match middle N components
        windows = row.unfold(0, self.n_middle_pattern_len, 1)
        matches = torch.zeros(windows.shape[0], dtype=torch.bool)
        for pattern in self.N_middle_torch:
            matches |= (windows == pattern).all(dim=1)
        l_offset = 50
        r_offset = 40
        offset = l_offset + r_offset
        for j in range(self.n_middle_pattern_len + offset):
            new_row[j:cols-self.n_middle_pattern_len-offset+j+1] = torch.where(
                matches[l_offset:-r_offset] & (new_row[j:cols-self.n_middle_pattern_len-offset+j+1] <= 1),
                new_row[j:cols-self.n_middle_pattern_len-offset+j+1] + 3,
                new_row[j:cols-self.n_middle_pattern_len-offset+j+1]
            )

        # Match middle Y components
        windows = row.unfold(0, self.y_middle_pattern_len, 1)
        matches = torch.zeros(windows.shape[0], dtype=torch.bool)
        for pattern in self.Y_middle_torch:
            matches |= (windows == pattern).all(dim=1)
        matches = matches & (new_row[:cols-self.y_middle_pattern_len+1] < 3)
        for j in range(self.y_middle_pattern_len + offset):
            new_row[j:cols-self.y_middle_pattern_len-offset+j+1] = torch.where(
                matches[l_offset:-r_offset] & (new_row[j:cols-self.y_middle_pattern_len-offset+j+1] == 1),
                5,
                new_row[j:cols-self.y_middle_pattern_len-offset+j+1]
            )

        # Match outer Y components
        rolled_row = torch.roll(row, -135)
        windows1 = row.unfold(0, self.y_outer_pattern_len, 1)
        windows2 = rolled_row.unfold(0, self.y_outer_pattern_len, 1)
        matches = torch.zeros(windows1.shape[0], dtype=torch.bool)
        for pattern1, pattern2 in self.Y_outer_torch:
            matches |= ((windows1 == pattern1).all(dim=1) & (windows2 == pattern2).all(dim=1))
        offset = 150
        for j in range(self.y_outer_pattern_len + offset):
            new_row[j:cols-self.y_outer_pattern_len-offset+j+1] = torch.where(
                matches[:-offset] & (new_row[j:cols-self.y_outer_pattern_len-offset+j+1] == 1),
                5,
                new_row[j:cols-self.y_outer_pattern_len-offset+j+1]
            )

        return new_row
 
    def find_leader(self):
        self.str_c = self.to_str(self.hidden_world)
        for l in self.strL:
            if l in self.str_c:
                return self.str_c.index(l)
        return None

    def decode_static_symbols(self, cg_config):
        dec = {4: "N", 5: "Y"}
        tape = ""
        largest_tape_index = -np.inf
        while (4 in cg_config) or (5 in cg_config):
            indices = torch.where((cg_config == 4) | (cg_config == 5))[0]
            tape_index = indices[-1].item()  # Get the last index
            tape+= dec[cg_config[tape_index].item()]
            cg_config = cg_config[:tape_index-200]
            largest_tape_index = max(largest_tape_index, tape_index)
        return tape, largest_tape_index

    def check_if_config_decodable(self, config):
        str_c = self.to_str(config)

        #find the first leader
        first_l = -1
        for s in self.strL:
            if s in str_c:
                first_l = str_c.index(s)

        if first_l == -1: #config is not decodable since there is not raw leader on the tape
            return None, None

        ossifier_index = -np.inf
        for s in self.strA4:
            j = str_c.rfind(s)
            if j>-1:
                ossifier_index = max(j+len(self.right_crop(s+self.str_ether)), ossifier_index)


        cg_config = self.coarse_grain(config[:first_l])
        static_tape_content, largest_tape_index = self.decode_static_symbols(cg_config)

        if np.abs(first_l-largest_tape_index)<300:
            return first_l, "tape"
        elif np.abs(first_l-ossifier_index)<300:
            return first_l, "oss"
        else:
            return None, None

    def tau(self):
        IT = 210
        it = 0

        init_time = self.time
        self.step(1000)

        while True:
            self.step(it = IT)
            it += IT
            leader_index, dec_type = self.check_if_config_decodable(self.hidden_world)
            if leader_index:
                break

        self.img_center = leader_index
        #print(self.time - init_time)
        if dec_type == "oss":
            print("no more data on the tape; halting")
            return False, None
        elif dec_type == "tape":
            return True, self.time - init_time

    def decode(self, right_start_read):
        right_crop = min(right_start_read, len(self.hidden_world))

        self.str_c = self.to_str(self.hidden_world[:right_crop])

        a4_index = -1
        for s in self.strA4:
            a4_index = max(a4_index, self.str_c.rfind(s))
        if a4_index == -1:
            raise ValueError("No ossifier found in the string.")

        last_ossifier_index = max(0, a4_index-200*14) #should not cut any ether in half

        str_c = self.str_c[:last_ossifier_index]

        a4_index = -1
        for s in self.strA4:
            a4_index = max(a4_index, str_c.rfind(s))
        if a4_index == -1:
            raise ValueError("No ossifier found in the string.")

        second_to_last_ossifier_index = max(0, a4_index-200*14) #should not cut any ether in half

        #print("second to last ossifier index: ", second_to_last_ossifier_index)

        decoder_window = self.hidden_world[second_to_last_ossifier_index:right_crop] #this is all the information the decoder has access to
        self.str_c = self.to_str(decoder_window)
        decoder_window = self.crop(self.str_c)
        self.str_c = self.to_str(decoder_window)
        
        #find the first leader
        first_l = -1
        for s in self.strL:
            first_l = max(first_l, self.str_c.rfind(s))
            if s in self.str_c:
                first_l = self.str_c.index(s)
        if first_l == -1:
            raise ValueError("No leader found in the string.")

        #print("first leader index: ", first_l)

        decoder_window = decoder_window[:first_l]
        self.str_c = self.to_str(decoder_window)
        self.hidden_world = torch.tensor(200*self.ether+decoder_window+200*self.ether, dtype=torch.int)
        
        ossifiers = []
        for o in self.strO:
            if o in self.str_c:
                ossifiers += [m.start() for m in re.finditer(o, self.str_c)]

        #print("ossifiers found: ", len(ossifiers))
        count = 0

        cropped_config = self.to_str(self.hidden_world)
        dist_to_hit_tape_data = np.inf

        while dist_to_hit_tape_data > 1000:
            count += 1
            #print(count)
            
            self.step(it = 1050, update = False)
            self.update_hidden_world(prolong_right = False)

            self.str_c = self.to_str(self.hidden_world)
            Es = []
            for e in self.strE:
                if e in self.str_c:
                    Es += [m.start() for m in re.finditer(e, self.str_c)]

            if not count%20:
                #print("Es found: ", len(Es))
                cg_config = self.coarse_grain(self.hidden_world)
                static_tape_content, largest_tape_index = self.decode_static_symbols(cg_config)
                #print("static tape content: ", static_tape_content[::-1]) 

            max_ossifier_index = -np.inf
            for s in self.strO:
                j = self.str_c.rfind(s)
                if j>-1:
                    max_ossifier_index = max(j+len(self.right_crop(s+self.str_ether)), max_ossifier_index)

            min_c2_index = np.inf
            for s in self.strC2:
                j = self.str_c.find(s)
                if j>-1:
                    min_c2_index = min(j, min_c2_index)

            dist_to_hit_tape_data = min_c2_index - max_ossifier_index

        return static_tape_content

    def step(self, it = 1, update = True):
        """
            Steps the automaton one timestep, recording the state of the world in self.world.
        """

        for t in range(it):
            indices = (torch.roll(self.hidden_world,shifts=(1))*4+self.hidden_world*2+torch.roll(self.hidden_world,shifts=(-1))*1) # (W,), compute in parallel all sums
            self.hidden_world=self.rule[indices] # This is the same as [ rule[indices[i]] for i in range(W)]
            self.time += 1
            if (not self.time%210) and update:
                self.update_hidden_world()
        
    def plot_worldmap(self, save_path = None, title = None):
        height, width, _ = self._worldmap.shape
        aspect_ratio = width / height
        plt.figure(figsize=(aspect_ratio * 10, 10))  # Adjust the scaling factor (e.g., 10) as needed
        plt.imshow(self._worldmap)
        plt.axis('off')
        if title:
            plt.title(title, fontsize=20)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

    def draw(self, iterations, save_path = None, title = None):
        self._worldmap = torch.zeros((iterations, self.w, 3), dtype=torch.float32)
        self.worlds = torch.zeros((iterations, self.w+2*self.offset), dtype=torch.int) # Vector of (render_size, W) elements

        self.init_hidden_world = self.hidden_world.clone()
        self.init_time = self.time

        l = self.find_leader()
        if not l:
            print("no leader found")
            self.img_center = self.world_center
        else:
            self.img_center = l


        for i in range(iterations):
            self.step(update = False)
            self.world = self.hidden_world[self.img_center-int(self.w//2)-self.offset:self.img_center+int(self.w//2)+self.offset]
            self.worlds[i, :] = self.world
            
        
        #self.worlds = self.coarse_grain(self.worlds)

        for i in range(iterations):
            self._worldmap[i, :, :] = self.colors[self.worlds[i, self.offset:-self.offset]]

        self.plot_worldmap(save_path, title = None)

        self.hidden_world = self.init_hidden_world.clone()
        self.time = self.init_time



def test_decoder(img_size, tape, cyclic_tag, IT = 10):
    folder = tape+"_"+"-".join(cyclic_tag)
    if not os.path.exists(folder):
        os.makedirs(folder)

    auto = Rule110Universality((img_size), tape, cyclic_tag)
    auto.get_init_hidden_world()

    save_path=f"{folder}/{str(0)}_{tape[::-1]}.png"
    auto.draw(2000, save_path, title=tape)

    print("total length of encoded cyclic tag: ", auto.len_cyclic_tag)

    right_start_read = auto.zero_index+2*auto.long_ossifier_distance*len(tape)+auto.len_cyclic_tag

    for it in range(IT):
        init_zero_index = auto.zero_index
        init_world_center = auto.world_center
        next_it, t = auto.tau()
        diff = auto.zero_index - init_zero_index
        right_start_read += diff
        print(f"iteration: {it}, sim time: {t}, config size: {auto.hidden_world.shape[0]}, internally, zero index was shifted by {diff}")
        
        if not next_it:
            print("no more data on the tape; halting")
            return
        
        init_oss_index = auto.oss_index
        init_leader_index = auto.last_e_glider_index
        init_hidden_world = auto.hidden_world.clone()
        init_time = auto.time


        

        tape_content = auto.decode(right_start_read)
        save_path=f"{folder}/{str(it+1)}_{tape_content}.png"
        right_start_read = right_start_read - int(np.round(t*auto.e_speed))+int(auto.len_cyclic_tag//len(auto.cyclic_tag))

        auto.hidden_world = init_hidden_world.clone()
        auto.oss_index = init_oss_index
        auto.last_e_glider_index = init_leader_index
        auto.time = init_time

        l = auto.find_leader()

        print("buffer for decoding window: ", right_start_read-l)

        #if not l:
        #    print("auto no leader found")
        #else:
        #    print("auto leader found: ", l)

        auto.draw(2000, save_path, title=tape_content)


        print()
        print() 


img_size = 4500
tape = "YN"
cyclic_tag = ["YYNYNY", "NYNNYNNNYNNY", "NNNYNY", "NNNYYY"]

test_decoder(img_size, tape, cyclic_tag, IT = 50)

