#!/usr/bin/env python
# Getting the GUI stuff
import wx
import wx.lib.dialogs

# Optimazation files
from scipy.optimize import *
from numpy import *
import KTKQ_Evaluation23

# Handling files and directories
import os

class TextFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'Wageningen B-Series', pos=(0,0),
                         size=(450, 750))
        self.filename = ""
        
        #Create Status bar
        self.CreateStatusBar()
        
        # Begin Toolbar / Menubar
        menubar = wx.MenuBar()
        file = wx.Menu()
        file.Append(-1, '&New')
        menuItem1 = file.Append(-1, "&Save")
        menuItem2 = file.Append(-1, "&Quit")
        file.AppendSeparator()
        helpMenu = wx.Menu()
        menuItem3 = helpMenu.Append(-1, 'Design Tasks')
        menuItem4 = helpMenu.Append(-1, '&Directions')
        menubar.Append(file, '&File')
        menubar.Append(helpMenu, '&About')
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.OnSave, menuItem1)
        self.Bind(wx.EVT_MENU, self.OnCloseMe, menuItem2)
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        self.Bind(wx.EVT_MENU, self.OnMethods, menuItem3)
        self.Bind(wx.EVT_MENU, self.OnAbout, menuItem4)
        
        # Begin List of Input lines
        panel = wx.Panel(self, -1)
        Input0Label = wx.StaticText(panel, -1, " Diameter [feet]:")
        self.Input0Text = wx.TextCtrl(panel, -1, "18.0", size=(175, -1))
        self.Input0Text.SetInsertionPoint(0)
        
        Input1Label = wx.StaticText(panel, -1, " Operating RPM [RPM]:")
        self.Input1Text = wx.TextCtrl(panel, -1, "77.0", size=(175, -1))
        self.Input1Text.SetInsertionPoint(0)
        
        Input2Label = wx.StaticText(panel, -1, " Total Resistance [lbs]:")
        self.Input2Text = wx.TextCtrl(panel, -1, "61900.0", size=(175, -1))
        self.Input2Text.SetInsertionPoint(0)
        
        Input3Label = wx.StaticText(panel, -1, " Number of Blades [-]:")
        self.Input3Text = wx.TextCtrl(panel, -1, "5.", size=(175, -1))
        self.Input3Text.SetInsertionPoint(0)
        
        Input4Label = wx.StaticText(panel, -1, " Delivered Horsepower [HP]:")
        self.Input4Text = wx.TextCtrl(panel, -1, "3832.0", size=(175, -1))
        self.Input4Text.SetInsertionPoint(0)
        
        Input5Label = wx.StaticText(panel, -1, " Speed [knots]:")
        self.Input5Text = wx.TextCtrl(panel, -1, "16.0", size=(175, -1))
        self.Input5Text.SetInsertionPoint(0)
        
        Input6Label = wx.StaticText(panel, -1, " Wake fraction [-]:")
        self.Input6Text = wx.TextCtrl(panel, -1, "0.252", size=(175, -1))
        self.Input6Text.SetInsertionPoint(0)
        
        Input7Label = wx.StaticText(panel, -1, " Thrust Deduction [-]:")
        self.Input7Text = wx.TextCtrl(panel, -1, "0.155", size=(175, -1))
        self.Input7Text.SetInsertionPoint(0)
        
        Input8Label = wx.StaticText(panel, -1, " Minimum Pitch Ratio [-]:")
        self.Input8Text = wx.TextCtrl(panel, -1, "0.5", size=(175, -1))
        self.Input8Text.SetInsertionPoint(0)
        
        Input9Label = wx.StaticText(panel, -1, " Maximum Pitch Ratio [-]:")
        self.Input9Text = wx.TextCtrl(panel, -1, "1.4", size=(175, -1))
        self.Input9Text.SetInsertionPoint(0)
        
        Input10Label = wx.StaticText(panel, -1, " Minimum Coefficient of Advance [-]:")
        self.Input10Text = wx.TextCtrl(panel, -1, "0.0", size=(175, -1))
        self.Input10Text.SetInsertionPoint(0)
        
        Input11Label = wx.StaticText(panel, -1, " Maximum Coefficient of Advance [-]:")
        self.Input11Text = wx.TextCtrl(panel, -1, "1.6", size=(175, -1))
        self.Input11Text.SetInsertionPoint(0)
        
        Input12Label = wx.StaticText(panel, -1, " Submergence of Propeller Shaft [feet]:")
        self.Input12Text = wx.TextCtrl(panel, -1, "38.6", size=(175, -1))
        self.Input12Text.SetInsertionPoint(0)
        
        Input13Label = wx.StaticText(panel, -1, " Minimum EAR [-]:")
        self.Input13Text = wx.TextCtrl(panel, -1, "0.3", size=(175, -1))
        self.Input13Text.SetInsertionPoint(0)
        
        Input14Label = wx.StaticText(panel, -1, " Maximum EAR [-]:")
        self.Input14Text = wx.TextCtrl(panel, -1, "1.05", size=(175, -1))
        self.Input14Text.SetInsertionPoint(0)
        
        Input15Label = wx.StaticText(panel, -1, " Gravitational Acceleration [ft/s^2]:")
        self.Input15Text = wx.TextCtrl(panel, -1, "32.174", size=(175, -1))
        self.Input15Text.SetInsertionPoint(0)
        
        Input16Label = wx.StaticText(panel, -1, " Temperature [F]:")
        self.Input16Text = wx.TextCtrl(panel, -1, "59.", size=(175, -1))
        self.Input16Text.SetInsertionPoint(0)
        
        Input17Label = wx.StaticText(panel, -1, " Relative Rotation Efficiency [-]:")
        self.Input17Text = wx.TextCtrl(panel, -1, "1.018", size=(175, -1))
        self.Input17Text.SetInsertionPoint(0)
        
        sizer = wx.FlexGridSizer(cols=2, hgap=6, vgap=6)
        sizer.AddMany([Input0Label, self.Input0Text, Input1Label, self.Input1Text,\
                      Input2Label, self.Input2Text, Input3Label, self.Input3Text,\
                      Input4Label, self.Input4Text, Input5Label, self.Input5Text,\
                      Input6Label, self.Input6Text, Input7Label, self.Input7Text,\
                      Input8Label, self.Input8Text, Input9Label, self.Input9Text,\
                      Input10Label, self.Input10Text, Input11Label, self.Input11Text,\
                      Input12Label, self.Input12Text, Input13Label, self.Input13Text,\
                      Input14Label, self.Input14Text, Input15Label, self.Input15Text,\
                      Input16Label, self.Input16Text, Input17Label, self.Input17Text])
        panel.SetSizer(sizer)
        
        # Create Check boxes
        List = ['Design Task 1', 'Design Task 2', 'Design Task 3', 'Design Task 4']
        self.listbox = wx.ListBox(panel, -1, (55, 525), wx.DefaultSize,
                                  List, wx.LB_SINGLE)
        self.Bind(wx.EVT_LISTBOX, self.OnList, self.listbox)
        
        # dirname is an APPLICATION variable that we're choosing to store
        # in with the frame - it's the parent directory for any file we
        # choose to edit in this frame
        self.dirname = ''
        
        # Begin buttons
        button = wx.Button(panel, label="Close", pos=(195,525), size=(50,25))
        self.Bind(wx.EVT_BUTTON, self.OnCloseMe, button)
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
        
        button1 = wx.Button(panel, label="Save", pos=(245,525), size=(50,25))
        self.Bind(wx.EVT_BUTTON, self.OnSave, button1)
        
        button2 = wx.Button(panel, label="Run", pos=(295,525), size=(50,25))
        self.Bind(wx.EVT_BUTTON, self.OnRun, button2)
        
        button3 = wx.Button(panel, label="View Results", pos=(195,550), size=(150,25))
        self.Bind(wx.EVT_BUTTON, self.OnResults, button3)
    
    def OnList(self, event):
        for eachText11 in [self.Input0Text, self.Input1Text,\
                          self.Input2Text, self.Input3Text, self.Input4Text,\
                          self.Input5Text,\
                          self.Input6Text, self.Input7Text, self.Input8Text, \
                          self.Input9Text,\
                          self.Input10Text, self.Input11Text, self.Input12Text,\
                          self.Input13Text,\
                          self.Input14Text, self.Input15Text, self.Input16Text, \
                          self.Input17Text]:
            eachText11.Enable(True)
        self.lselect1 = self.listbox.GetSelection()
        if self.lselect1 == 0:
            for eachText1 in [self.Input1Text, self.Input4Text, self.Input17Text]:
                eachText1.Enable(False)
            self.methodnumber = "method_1"
        elif self.lselect1 == 1:
            for eachText2 in [self.Input1Text, self.Input2Text]:
                eachText2.Enable(False)
            self.methodnumber = "method_2"
        elif self.lselect1 == 2:
            for eachText3 in [self.Input0Text, self.Input4Text, self.Input17Text]:
                eachText3.Enable(False)
            self.methodnumber = "method_3"
        elif self.lselect1 == 3:
            for eachText4 in [self.Input0Text, self.Input2Text]:
                eachText4.Enable(False)
            self.methodnumber = "method_4"
        return self.methodnumber, self.lselect1
    
    # Event for closing
    def OnCloseMe(self, event):
        self.Close(True)
    
    def OnCloseWindow(self, event):
        self.Destroy()
    
    def OnMethods(self, event):
        methods ="""Design Task 1 = Given RT, D, V. Find optimum Revolutions \
Per Minute - RPMopt\
\
Design Task 2 = Given Pd, D, V. Find optimum RPM - RPMopt
\
\
Design Task 3 = Given RT, N, V. Find optimum Diameter - Dopt
\
\
Design Task 4 = Given Pd, N, V. Find optimum Diameter - Dopt"""
        dialog1 = wx.lib.dialogs.ScrolledMessageDialog(self, methods, "Design Task Options")
        dialog1.ShowModal()
    
    def OnAbout(self, event):
        directions ="""This program is pretty straight forward and can be completed\
in a few simply steps.\
1.) Select the desired calculation task and insert your inputs\
into the provided spaces.\
2.) Save the file. The program names the file based on the selected List\
Item. Just save, do not change the file name.\
\
3.) After the file is saved, select the 'RUN' button.\
\
4.) Select "View Results" button and the optimum values will \
be displayed."""
        dialog = wx.lib.dialogs.ScrolledMessageDialog(self, directions, "Directions")
        dialog.ShowModal()
    
    # Event for saving
    def OnSave(self,e):
        #Specify values to save for list item selected
        if self.lselect1 == 0:
            v0 = float(self.Input0Text.GetValue())
            v2 = float(self.Input2Text.GetValue())
            v3 = float(self.Input3Text.GetValue())
            v5 = float(self.Input5Text.GetValue())
            v6 = float(self.Input6Text.GetValue())
            v7 = float(self.Input7Text.GetValue())
            v8 = float(self.Input8Text.GetValue())
            v9 = float(self.Input9Text.GetValue())
            v10 = float(self.Input10Text.GetValue())
            v11 = float(self.Input11Text.GetValue())
            v12 = float(self.Input12Text.GetValue())
            v13 = float(self.Input13Text.GetValue())
            v14 = float(self.Input14Text.GetValue())
            v15 = float(self.Input15Text.GetValue())
            v16 = float(self.Input16Text.GetValue())
            self.values = v0, v2, v3, v5, v6, v7, v8, v9, v10, v11,\
                         v12, v13, v14, v15, v16
        elif self.lselect1 == 1:
            v0 = float(self.Input0Text.GetValue())
            v3 = float(self.Input3Text.GetValue())
            v4 = float(self.Input4Text.GetValue())
            v5 = float(self.Input5Text.GetValue())
            v6 = float(self.Input6Text.GetValue())
            v7 = float(self.Input7Text.GetValue())
            v8 = float(self.Input8Text.GetValue())
            v9 = float(self.Input9Text.GetValue())
            v10 = float(self.Input10Text.GetValue())
            v11 = float(self.Input11Text.GetValue())
            v12 = float(self.Input12Text.GetValue())
            v13 = float(self.Input13Text.GetValue())
            v14 = float(self.Input14Text.GetValue())
            v15 = float(self.Input15Text.GetValue())
            v16 = float(self.Input16Text.GetValue())
            v17 = float(self.Input17Text.GetValue())
            self.values = v0, v3, v4, v5, v6, v7, v8, v9, v10, v11,\
                         v12, v13, v14, v15, v16, v17
        elif self.lselect1 == 2:
            for eachText3 in [self.Input0Text, self.Input4Text, self.Input17Text]:
                eachText3.Enable(False)
            v1 = float(self.Input1Text.GetValue())
            v2 = float(self.Input2Text.GetValue())
            v3 = float(self.Input3Text.GetValue())
            v5 = float(self.Input5Text.GetValue())
            v6 = float(self.Input6Text.GetValue())
            v7 = float(self.Input7Text.GetValue())
            v8 = float(self.Input8Text.GetValue())
            v9 = float(self.Input9Text.GetValue())
            v10 = float(self.Input10Text.GetValue())
            v11 = float(self.Input11Text.GetValue())
            v12 = float(self.Input12Text.GetValue())
            v13 = float(self.Input13Text.GetValue())
            v14 = float(self.Input14Text.GetValue())
            v15 = float(self.Input15Text.GetValue())
            v16 = float(self.Input16Text.GetValue())
            self.values = v1, v2, v3, v5, v6, v7, v8, v9, v10, v11,\
                         v12, v13, v14, v15, v16
        elif self.lselect1 == 3:
            v1 = float(self.Input1Text.GetValue())
            v3 = float(self.Input3Text.GetValue())
            v4 = float(self.Input4Text.GetValue())
            v5 = float(self.Input5Text.GetValue())
            v6 = float(self.Input6Text.GetValue())
            v7 = float(self.Input7Text.GetValue())
            v8 = float(self.Input8Text.GetValue())
            v9 = float(self.Input9Text.GetValue())
            v10 = float(self.Input10Text.GetValue())
            v11 = float(self.Input11Text.GetValue())
            v12 = float(self.Input12Text.GetValue())
            v13 = float(self.Input13Text.GetValue())
            v14 = float(self.Input14Text.GetValue())
            v15 = float(self.Input15Text.GetValue())
            v16 = float(self.Input16Text.GetValue())
            v17 = float(self.Input17Text.GetValue())
            self.values = v1, v3, v4, v5, v6, v7, v8, v9, v10, v11,\
                         v12, v13, v14, v15, v16, v17
        
        self.sv = asarray(self.values)
        
        # Save away the edited text
        # Open the file, do an RU sure check for an overwrite!
        dlg = wx.FileDialog(self, "Choose a file", self.dirname, self.methodnumber, \
                           ".txt", \
                           wx.SAVE | wx.OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            # Grab the content to be saved
            # Open the file for write, write, close
            self.filename=dlg.GetFilename()
            self.dirname=dlg.GetDirectory()
            filehandle=open(os.path.join(self.dirname, self.filename),"w")
            for i in range(len(self.sv)):
                filehandle.write("%f " %(self.sv[i]))
            filehandle.close()
        # Get rid of the dialog to keep things tidy
        dlg.Destroy()
    
    ###############################################################################
    def OnRun(self, event):
        if self.lselect1 == 0:
            # Our given variables will be imported from a set-up set: method1.txt
            def objmethod1(xfree, args = None):
                f = open("method_1.txt")
                lines = f.readlines()
                f.close()
                
                # Make data from input file into usable data
                for line in lines:
                    strlist = line.split()
                    D = float(strlist[0]) # Diameter [Feet]
                    RT = float(strlist[1]) # Resistance [lbs]
                    Z = float(strlist[2]) # Number of Blades
                    V = float(strlist[3]) # Ship Speed [knots]
                    w = float(strlist[4]) # Wake Fraction \
                                         # Typical value for Sub Cavitating Prop w = 0.1 - 0.3
                    t = float(strlist[5]) # Thrust Deduction \
                                         # Typical value for Sub Cavitating Prop t = 0.05 - 0.1
                    PDmin = float(strlist[6]) # Minimum Pitch Ratio \
                                             # Constraint
                    PDmax = float(strlist[7]) # Maximum Pitch Ratio \
                                             # Constraint
                    Jmin = float(strlist[8]) # Minimum Advance Coefficient\
                                            # Constraint
                    Jmax = float(strlist[9]) # Maximum Advance Coefficient\
                                            # Constraint
                    h = float(strlist[10]) # Distance to propeller shaft\
                                          # centerline
                    EARmin = float(strlist[11]) # min EAR
                    EARmax = float(strlist[12]) # max EAR
                    g = float(strlist[13]) # gravitational acceleration
                    T = float(strlist[14]) # temperature
                
                V1 = V*1.688 # convert speed from knots to fps
                
                J = xfree[0]
                PD = xfree[1]
                EAR = xfree[2]
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-\
                     0.0001*(T**2)+0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+\
                    2e-08*(T**2)-1e-06*T + 4e-05
                
                # Calculated data for givens:
                VA = V1*(1-w)
                EHP = RT* V1
                n = VA / (xfree[0]* D)
                
                # Compute chord length
                CR = 2.073*xfree[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12 # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xfree[0], xfree[1], xfree[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xfree[0], xfree[1], xfree[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                # Calculate Thrust and Torque
                Th = KT*rho*(n**2)*(D**4)
                Q = KQ*rho*(n**2)*(D**5)
                
                CT = EHP / (rho*(1-t)*((1-w)**2)*(V1**3)*(D**2))
                
                pai = 14.7 # [psi]
                pa = pai*144 # [psf]
                pvi = (2e-11)*(T**5) - (1e-9)*(T**4) + (1e-6)*T**3 - (7e-5)*(T**2)+ \
                     0.0061*T - 0.0726 # [psi]
                pv = pvi*144 # [psf]
                
                #Th = KT*rho*(n**2)*(D**4) # [lbs]
                p0 = pa + (rho * g * h) # [psf]
                
                # k = 0 for normal vessels and k = 0.2 for single screw, high loading
                k = 0.2
                EARc = (((1.3 + 0.3 * Z)*Th)/((p0-pv)*((D)**2)))+k
                
                p07 = pa + (rho * g * h)
                VA2 = (VA**2)
                VA22 = (pi*0.7*D*n)**2
                v1 = sqrt(VA2 + VA22)
                q07 = (0.5*rho*(v1**2))
                sigma07 = (p07-pv)/q07
                tau = 0.3*sqrt(sigma07) - 0.03
                Ap = Th / (q07 * tau)
                Ad = Ap / (1.067-0.229 * xfree[1])
                Ao = pi*(D**2)/4
                EARb = Ad/Ao
                
                # Our given range constants:
                # remain in these ranges
                # 2 <= Z <= 7
                # 0.3 <= EAR <= 1.05
                # 0.5 <= PD <= 1.4
                # Outside these ranges the values become unrealiable
                
                # Set-up Constraints for Propellers
                constrvalue = []
                
                g1 = xfree[1] - PDmin
                constrvalue.append(g1)
                
                g2 = PDmax - xfree[1]
                constrvalue.append(g2)
                
                g3 = xfree[0] - Jmin
                constrvalue.append(g3)
                
                g4 = Jmax - xfree[0]
                constrvalue.append(g4)
                
                g5 = KT - (CT * xfree[0]**2)
                constrvalue.append(g5)
                
                g6 = xfree[2] - EARmin
                constrvalue.append(g6)
                
                g7 = EARmax - xfree[2]
                constrvalue.append(g7)
                
                g8 = xfree[2] - EARc
                constrvalue.append(g8)
                
                g9 = xfree[2] - EARb
                constrvalue.append(g9)
                
                objmethod1 = -(xfree[0]* KT / (2*pi*KQ))
                # print objmethod1, EARb, EARc
                # print objmethod1
                # print xfree
                # print topt, tmin, BSHP
                
                for constr in constrvalue:
                    if constr<0:
                        objmethod1 = objmethod1 + 11.0 * ((constr**2))
                
                return objmethod1
            
            # Begin Optimization
            if __name__ == "__main__":
                x0 = [ .7, .7, .7 ]
                xsolution, fopt, niter, ncalls, error \
                    = fmin(objmethod1, x0, xtol=1e-8, full_output=1, disp=0)
                xs = abs(fopt)
                # print xsolution, xs, niter, ncalls, error
                print "J =", xsolution[0]
                print "PD =", xsolution[1]
                print "EAR =", xsolution[2]
                print "Max Efficency =", xs
                
                f = open("method_1.txt")
                lines = f.readlines()
                f.close()
                
                for line in lines:
                    strlist = line.split()
                    D = float(strlist[0]) # Diameter [Feet]
                    Z = float(strlist[2])
                    V = float(strlist[3]) # Ship Speed [knots]
                    w = float(strlist[4]) # Wake Fraction Typical \
                                         # value for Sub Cavitating Prop w = 0.1 - 0.3
                    h = float(strlist[10]) # Distance to propeller shaft \
                                          # centerline
                    T = float(strlist[14])
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*\
                     (T**2)+0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*\
                    (T**2)-1e-06*T + 4e-05
                
                V1 = V*1.688 # convert speed from knots to fps
                VA = V1*(1-w)
                n = VA / (xsolution[0]* D)
                
                print "opt RPM =", n*60, "[rpm]"
                
                # Compute chord length
                CR = 2.073*xsolution[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12. # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xsolution[0], xsolution[1], xsolution[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xsolution[0], xsolution[1], xsolution[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                # Calculate Thrust and Torque
                Th = KT*rho*(n**2)*(D**4)
                Q = KQ*rho*(n**2)*(D**5)
                
                self.Jopt = xsolution[0]
                self.PDopt = xsolution[1]
                self.EARopt = xsolution[2]
                self.maxeff = xs
                self.rpm = n*60
                self.Thrust = Th
                self.Torque = Q
        
        ################################################################################
        elif self.lselect1 == 1:
            # Our given variables will be imported from a set-up set: method2.txt
            def objmethod2(xfree, args = None):
                f = open("method_2.txt")
                lines = f.readlines()
                f.close()
                
                # Make data from input file into usable data
                for line in lines:
                    strlist = line.split()
                    D = float(strlist[0]) # Diameter [Feet]
                    Z = float(strlist[1]) # Number of Blades
                    DHP = float(strlist[2]) # Delivered Horse Power [HP]
                    V = float(strlist[3]) # Ship Speed [knots]
                    w = float(strlist[4]) # Wake Fraction Typical\
                                         # value for Sub Cavitating Prop w = 0.1 - 0.3
                    t = float(strlist[5]) # Thrust Deduction Typical\
                                         # value for Sub Cavitating Prop t = 0.05 - 0.1
                    PDmin = float(strlist[6]) # Minimum Pitch Ratio \
                                             # Constraint
                    PDmax = float(strlist[7]) # Maximum Pitch Ratio \
                                             # Constraint
                    Jmin = float(strlist[8]) # Minimum Advance Coefficient\
                                            # Constraint
                    Jmax = float(strlist[9]) # Maximum Advance Coefficient\
                                            # Constraint
                    h = float(strlist[10]) # Distance to propeller shaft \
                                          # centerline
                    EARmin = float(strlist[11]) # min EAR
                    EARmax = float(strlist[12]) # max EAR
                    g = float(strlist[13]) # gravitational acceleration
                    T = float(strlist[14]) # temperature
                    nr = float(strlist[15]) # Relative Rotation Efficiency
                
                V1 = V*1.688 # convert speed from knots to fps
                DHP1 = DHP*550 # convert hp to foot-lbs/second
                
                J = xfree[0]
                PD = xfree[1]
                EAR = xfree[2]
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*\
                     (T**2)+0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*\
                    (T**2)-1e-06*T + 4e-05
                
                # Calculated data for givens:
                VA = V1*(1-w)
                n = xfree[0]* D / VA
                
                # Compute chord length
                CR = 2.073*xfree[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12 # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xfree[0], xfree[1], xfree[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xfree[0], xfree[1], xfree[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                # T = KT*rho[0]*(n**2)*(D[0]**4)
                # Q = KQ*rho[0]*(n**2)*(D[0]**5)
                
                CQ = (DHP1*nr) / (2*pi*rho*((1-w)**3)*(V1**3)*(D**2))
                
                pai = 14.7 # [psi]
                pa = pai*144 # [psf]
                pvi = (2.e-11)*(T**5) - (1.e-9)*(T**4) + (1.e-6)*T**3 - (7.e-5)*(T**2)\
                     + 0.0061*T - 0.0726 # [psi]
                pv = pvi*144 # [psf]
                
                Th = KT*rho*(n**2)*(D**4) # [lbs]
                p0 = pa + (rho * g * h) # [psf]
                
                # k = 0 for normal vessels and k = 0.2 for single screw, high loading
                k = 0.2
                EARc = (((1.3 + 0.3 * Z)*Th)/((p0-pv)*((D)**2)))+k
                
                p07 = pa + (rho * g * h)
                VA2 = (VA**2)
                VA22 = (pi*0.7*D*n)**2
                v1 = sqrt(VA2 + VA22)
                q07 = (0.5*rho*(v1**2))
                sigma07 = (p07-pv)/q07
                tau = 0.3*sqrt(sigma07) - 0.03
                Ap = Th / (q07 * tau)
                Ad = Ap / (1.067-0.229 * xfree[1])
                Ao = pi*(D**2)/4
                EARb = Ad/Ao
                
                # Our given range constants:
                # remain in these ranges
                # 2 <= Z <= 7
                # 0.3 <= EAR <= 1.05
                # 0.5 <= PD <= 1.4
                # Outside these ranges the values become unrealiable
                
                # Set-up Constraints for Propellers
                constrvalue = []
                
                g1 = xfree[1] - PDmin
                constrvalue.append(g1)
                
                g2 = PDmax - xfree[1]
                constrvalue.append(g2)
                
                g3 = xfree[2] - EARmin
                constrvalue.append(g3)
                
                g4 = EARmax - xfree[2]
                constrvalue.append(g4)
                
                g5 = xfree[0] - Jmin
                constrvalue.append(g5)
                
                g6 = Jmax - xfree[0]
                constrvalue.append(g6)
                
                g7 = KQ - (CQ * xfree[0]**3)
                constrvalue.append(g7)
                
                g8 = xfree[2] - EARc
                constrvalue.append(g8)
                
                g9 = xfree[2] - EARb
                constrvalue.append(g9)
                
                objmethod2 = -(xfree[0]* KT / (2*pi*KQ))
                # print objmethod2
                # print xfree, CQ
                
                for constr in constrvalue:
                    if constr<0:
                        objmethod2 = objmethod2 + 122.8 * ((constr**2))
                
                return objmethod2
            
            # Begin Optimization
            if __name__ == "__main__":
                x0 = [ .6, .6, .6]
                xsolution, fopt, niter, ncalls, error \
                    = fmin(objmethod2, x0, xtol = 1e-08, full_output=1, disp=0)
                xs = abs(fopt)
                # print xsolution, fopt, niter, ncalls, error
                print "J =", xsolution[0]
                print "PD =", xsolution[1]
                print "EAR =", xsolution[2]
                print "max NO =", xs
                
                f = open("method_2.txt")
                lines = f.readlines()
                f.close()
                
                for line in lines:
                    strlist = line.split()
                    D = float(strlist[0]) # Diameter [Feet]
                    Z = float(strlist[1]) # Number of Blades
                    V = float(strlist[3]) # Ship Speed [knots]
                    w = float(strlist[4]) # Wake Fraction Typical \
                                         # value for Sub Cavitating Prop w = 0.1 - 0.3
                    h = float(strlist[10]) # Distance to propeller shaft centerline
                    T = float(strlist[14]) # temperature
                
                V1 = V*1.688 # convert speed from knots to fps
                VA = V1*(1-w)
                n = VA / (xsolution[0]* D)
                print "opt RPM =", n*60, "[rpm]"
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*(T**2)+\
                     0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*(T**2)-\
                    1e-06*T + 4e-05
                
                # Compute chord length
                CR = 2.073*xsolution[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12. # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xsolution[0], xsolution[1], xsolution[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xsolution[0], xsolution[1], xsolution[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                # Calculate Thrust and Torque
                Th = KT*rho*(n**2)*(D**4)
                Q = KQ*rho*(n**2)*(D**5)
                
                self.Jopt = xsolution[0]
                self.PDopt = xsolution[1]
                self.EARopt = xsolution[2]
                self.maxeff = xs
                self.rpm = n*60
                self.Thrust = Th
                self.Torque = Q
        
        ################################################################################################################
        elif self.lselect1 == 2:
            # Our given variables will be imported from a set-up set: method3.txt
            def objmethod3(xfree, args = None):
                f = open("method_3.txt")
                lines = f.readlines()
                f.close()
                
                # Make data from input file into usable data
                for line in lines:
                    strlist = line.split()
                    nrpm = float(strlist[0]) # Operating RPM
                    RT = float(strlist[1]) # Resistance [lbs]
                    Z = float(strlist[2]) # Number of Blades
                    V = float(strlist[3]) # Ship Speed [knots]
                    w = float(strlist[4]) # Wake Fraction Typical\
                                         # value for Sub Cavitating Prop w = 0.1 - 0.3
                    t = float(strlist[5]) # Thrust Deduction Typical\
                                         # value for Sub Cavitating Prop t = 0.05 - 0.1
                    PDmin = float(strlist[6]) # Minimum Pitch Ratio \
                                             # Constraint
                    PDmax = float(strlist[7]) # Maximum Pitch Ratio \
                                             # Constraint
                    Jmin = float(strlist[8]) # Minimum Advance Coefficient \
                                            # Constraint
                    Jmax = float(strlist[9]) # Maximum Advance Coefficient \
                                            # Constraint
                    h = float(strlist[10]) # Distance to propeller shaft \
                                          # centerline
                    EARmin = float(strlist[11]) # min EAR
                    EARmax = float(strlist[12]) # max EAR
                    g = float(strlist[13]) # Gravitational acceleration
                    T = float(strlist[14]) # Temperature
                
                V1 = V*1.688 # convert speed from knots to fps
                
                J = xfree[0]
                PD = xfree[1]
                EAR = xfree[2]
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*(T**2)+\
                     0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*(T**2)-\
                    1e-06*T + 4e-05
                
                n = nrpm/60 # convert rpm to rps
                
                # Calculated data for givens:
                VA = V1 * (1-w)
                D = VA / (xfree[0] * n)
                
                # Compute chord length
                CR = 2.073*xfree[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12 # Convert to inches
                
                #compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xfree[0], xfree[1], xfree[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xfree[0], xfree[1], xfree[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                #Calculate Thrust and Torque
                # T = KT*rho[0]*(n**2)*(D[0]**4)
                # Q = KQ*rho[0]*(n**2)*(D[0]**5)
                
                CT = (RT*n**2) / (rho*(1-t)*((1-w)**4)*(V1**4))
                
                pai = 14.7 # [psi]
                pa = pai*144 # [psf]
                pvi = (2.e-11)*(T**5) - (1.e-9)*(T**4) + (1.e-6)*T**3 - (7.e-5)*(T**2)\
                     + 0.0061*T - 0.0726 # [psi]
                pv = pvi*144 # [psf]
                
                Th = KT*rho*(n**2)*(D**4) # [lbs]
                p0 = pa + (rho * g * h) # [psf]
                
                # k = 0 for normal vessels and k = 0.2 for single screw, high loading
                k = 0.2
                EARc = (((1.3 + 0.3 * Z) * Th) / ((p0-pv)*((D)**2))) + k
                
                p07 = pa + (rho * g * h)
                VA2 = (VA**2)
                VA22 = (pi*0.7*D*n)**2
                v1 = sqrt(VA2 + VA22)
                q07 = (0.5*rho*(v1**2))
                sigma07 = (p07-pv)/q07
                tau = 0.3*sqrt(sigma07) - 0.03
                Ap = Th / (q07 * tau)
                Ad = Ap / (1.067-0.229 * xfree[1])
                Ao = pi*(D**2)/4
                EARb = Ad/Ao
                
                # Our given range constants:
                # remain in these ranges
                # 2 <= Z <= 7
                # 0.3 <= EAR <= 1.05
                # 0.5 <= PD <= 1.4
                # Outside these ranges the values become unrealiable
                
                # Set-up Constraints for Propellers
                constrvalue = []
                
                g1 = xfree[2] - EARmin
                constrvalue.append(g1)
                
                g2 = EARmax - xfree[2]
                constrvalue.append(g2)
                
                g3 = xfree[1] - PDmin
                constrvalue.append(g3)
                
                g4 = PDmax - xfree[1]
                constrvalue.append(g4)
                
                g5 = xfree[0] - Jmin
                constrvalue.append(g5)
                
                g6 = Jmax - xfree[0]
                constrvalue.append(g6)
                
                g7 = xfree[2] - EARc
                constrvalue.append(g7)
                
                g8 = xfree[2] - EARb
                constrvalue.append(g8)
                
                g9 = KT - (CT * xfree[0]**4)
                constrvalue.append(g9)
                
                objmethod3 = -(xfree[0]* KT / (2*pi*KQ))
                # print objmethod3
                # print xfree,CT
                
                for constr in constrvalue:
                    if constr<0:
                        objmethod3 = objmethod3 + 3.3 * ((constr**2))
                
                return objmethod3
            
            if __name__ == "__main__":
                x0 = [ .6, .6, .6]
                xsolution, fopt, niter, ncalls, error \
                    = fmin(objmethod3, x0, xtol = 1e-8, full_output=1, disp=0)
                xs = abs(fopt)
                # print xsolution, xs, niter, ncalls, error
                print "J =", xsolution[0]
                print "PD =", xsolution[1]
                print "EAR =", xsolution[2]
                print "max Efficiency =", xs
                
                f = open("method_3.txt")
                lines = f.readlines()
                f.close()
                
                for line in lines:
                    strlist = line.split()
                    nrpm = float(strlist[0])
                    Z = float(strlist[2]) # Number of Blades
                    V = float(strlist[3])
                    w = float(strlist[4])
                    h = float(strlist[10]) # Distance to propeller shaft \
                                          # centerline
                    T = float(strlist[14]) # Temperature
                
                n = nrpm/60
                V1 = V*1.688
                VA = V1 * (1-w)
                D = VA / (xsolution[0] * n)
                print "optimum diameter =", D, "[feet]"
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*(T**2)\
                     +0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*(T**2)\
                    -1e-06*T + 4e-05
                
                # Compute chord length
                CR = 2.073*xsolution[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12. # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xsolution[0], xsolution[1], xsolution[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xsolution[0], xsolution[1], xsolution[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                # Calculate Thrust and Torque
                Th = KT*rho*(n**2)*(D**4)
                Q = KQ*rho*(n**2)*(D**5)
                
                self.Jopt = xsolution[0]
                self.PDopt = xsolution[1]
                self.EARopt = xsolution[2]
                self.maxeff = xs
                self.D = D
                self.Thrust = Th
                self.Torque = Q
        
        ##################################################################################
        elif self.lselect1 == 3:
            # Our given variables will be imported from a set-up set: method4.txt
            def objmethod4(xfree, args = None):
                f = open("method_4.txt")
                lines = f.readlines()
                f.close()
                
                # Make data from input file into usable data
                for line in lines:
                    strlist = line.split()
                    nrpm = float(strlist[0]) # Operating RPM
                    Z = float(strlist[1]) # Number of Blades
                    DHP = float(strlist[2]) # Delivered Horse Power [HP]
                    V = float(strlist[3]) # Ship Speed [knots]
                    w = float(strlist[4]) # Wake Fraction Typical \
                                         # value for Sub Cavitating Prop w = 0.1 - 0.3
                    t = float(strlist[5]) # Thrust Deduction Typical \
                                         # value for Sub Cavitating Prop t = 0.05 - 0.1
                    PDmin = float(strlist[6]) # Minimum Pitch Ratio \
                                             # Constraint
                    PDmax = float(strlist[7]) # Maximum Pitch Ratio \
                                             # Constraint
                    Jmin = float(strlist[8]) # Minimum Advance Coefficient \
                                            # Constraint
                    Jmax = float(strlist[9]) # Maximum Advance Coefficient \
                                            # Constraint
                    h = float(strlist[10]) # Distance to propeller shaft \
                                          # centerline
                    EARmin = float(strlist[11]) # min EAR
                    EARmax = float(strlist[12]) # max EAR
                    g = float(strlist[13]) # Gravitational acceleration
                    T = float(strlist[14]) # Temperature
                    nr = float(strlist[15]) # Relative Rotation Efficiency
                
                V1 = V*1.688 # convert speed from knots to fps
                DHP1 = DHP*550 # convert hp to foot-lbs/second
                
                J = xfree[0]
                PD = xfree[1]
                EAR = xfree[2]
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*\
                     (T**2)+0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*\
                    (T**2)-1e-06*T + 4e-05
                
                n = nrpm/60
                
                # Calculated data for givens:
                VA = V1 * (1-w)
                D = VA / (xfree[0] * n)
                
                # Compute chord length
                CR = 2.073*xfree[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12 # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xfree[0], xfree[1], xfree[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xfree[0], xfree[1], xfree[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xfree[0], xfree[1], xfree[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                # T = KT*rho[0]*(n**2)*(D[0]**4)
                # Q = KQ*rho[0]*(n**2)*(D[0]**5)
                
                CQ = (DHP1*(n**2)*nr) / (2*pi*rho*((1-w)**5)*(V1**5))
                
                pai = 14.7 # [psi]
                pa = pai*144 # [psf]
                pvi = (2.e-11)*(T**5) - (1.e-9)*(T**4) + (1.e-6)*T**3 - (7.e-5)*(T**2)\
                     + 0.0061*T - 0.0726 # [psi]
                pv = pvi*144 # [psf]
                
                Th = KT*rho*(n**2)*(D**4) # [lbs]
                p0 = pa + (rho * g * h) # [psf]
                
                # k = 0 for normal vessels and k = 0.2 for single screw, high loading
                k = 0.2
                EARc = (((1.3 + 0.3 * Z) * Th) / ((p0-pv)*((D)**2))) + k
                
                p07 = pa + (rho * g * h)
                VA2 = (VA**2)
                VA22 = (pi*0.7*D*n)**2
                v1 = sqrt(VA2 + VA22)
                q07 = (0.5*rho*(v1**2))
                sigma07 = (p07-pv)/q07
                tau = 0.3*sqrt(sigma07) - 0.03
                Ap = Th / (q07 * tau)
                Ad = Ap / (1.067-0.229 * xfree[1])
                Ao = pi*(D**2)/4
                EARb = Ad/Ao
                
                # Our given range constants:
                # remain in these ranges
                # 2 <= Z <= 7
                # 0.3 <= EAR <= 1.05
                # 0.5 <= PD <= 1.4
                # Outside these ranges the values become unrealiable
                
                # Set-up Constraints for Propellers
                constrvalue = []
                
                g1 = xfree[2] - EARmin
                constrvalue.append(g1)
                
                g2 = EARmax - xfree[2]
                constrvalue.append(g2)
                
                g3 = xfree[1] - PDmin
                constrvalue.append(g3)
                
                g4 = PDmax - xfree[1]
                constrvalue.append(g4)
                
                g5 = xfree[0] - Jmin
                constrvalue.append(g5)
                
                g6 = Jmax - xfree[0]
                constrvalue.append(g6)
                
                g7 = xfree[2] - EARc
                constrvalue.append(g7)
                
                g8 = xfree[2] - EARb
                constrvalue.append(g8)
                
                g9 = KQ - (CQ * xfree[0]**5)
                constrvalue.append(g9)
                
                objmethod4 = -(xfree[0]* KT / (2*pi*KQ))
                # print objmethod4
                # print xfree,CT
                
                for constr in constrvalue:
                    if constr<0:
                        objmethod4 = objmethod4 + 44.5 * ((constr**2))
                
                return objmethod4
            
            if __name__ == "__main__":
                
                x0 = [ .7, .7, .7]
                xsolution, fopt, niter, ncalls, error \
                    = fmin(objmethod4, x0, xtol = 1e-8, full_output=1, disp=0)
                xs = abs(fopt)
                # print xsolution, xs, niter, ncalls, error
                print "J =", xsolution[0]
                print "PD =", xsolution[1]
                print "EAR =", xsolution[2]
                print "Max Efficiency =", xs
                
                f = open("method_4.txt")
                lines = f.readlines()
                f.close()
                
                for line in lines:
                    strlist = line.split()
                    nrpm = float(strlist[0])
                    Z = float(strlist[1]) # Number of Blades
                    V = float(strlist[3])
                    w = float(strlist[4])
                    h = float(strlist[10]) # Distance to propeller shaft \
                                          # centerline
                    T = float(strlist[14]) # Temperature
                
                n = nrpm/60
                V1 = V*1.688
                VA = V1 * (1.-w)
                D = VA / (xsolution[0] * n)
                print "optimum diameter =", D, "[feet]"
                
                # Density [lbs-sec^2/ft^4 - salt water, fresh = 1.94]
                rho = -5e-13*(T**6) + 2e-10*(T**5) -4e-08*(T**4)+3e-06*(T**3)-0.0001*\
                     (T**2)+0.003*T + 1.9679
                # Kinematic viscosity [ft^2/s]
                kv = 3e-17*(T**6) - 1e-14*(T**5) + 3e-12*(T**4)-3e-10*(T**3)+2e-08*\
                    (T**2)-1e-06*T + 4e-05
                
                # Calculated data for givens:
                VA = V1 * (1-w)
                D = VA / (xsolution[0] * n)
                
                # Compute chord length
                CR = 2.073*xsolution[2]*D/Z # Minimum chord length at R-0.75
                tc075 = (0.0185-0.00125*Z)*D/CR
                th = tc075 * CR * 12 # Convert to inches
                
                # compute Reynolds number at R-0.75
                RN = CR * ((VA**2 + ((0.75*pi*n*D)**2))**0.5) / kv
                
                KTC = KTKQ_Evaluation23.KTC(xsolution[0], xsolution[1], xsolution[2], Z)
                KQC = KTKQ_Evaluation23.KQC(xsolution[0], xsolution[1], xsolution[2], Z)
                DKT = KTKQ_Evaluation23.KTCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                DKQ = KTKQ_Evaluation23.KQCRC(xsolution[0], xsolution[1], xsolution[2], Z, RN)
                
                KT = KTC + DKT
                KQ = KQC + DKQ
                
                Th = KT*rho*(n**2)*(D**4)
                Q = KQ*rho*(n**2)*(D**5)
                
                self.Jopt = xsolution[0]
                self.PDopt = xsolution[1]
                self.EARopt = xsolution[2]
                self.maxeff = xs
                self.D = D
                self.Thrust = Th
                self.Torque = Q
    
    # Need some help here!
    def OnResults(self, event):
        if self.lselect1 == 0:
            results ="%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f" \
                    % ('J', self.Jopt, 'P/D', self.PDopt, 'EAR', self.EARopt,'Max Efficiency',\
                      self.maxeff, 'Optimum RPM', self.rpm, 'Thrust [lbs]', self.Thrust, \
                      'Torque [ft lb]', self.Torque )
            dialog = wx.MessageBox (results, 'Results', style = wx.OK )
        elif self.lselect1 == 1:
            results1 ="%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f" \
                     % ('J', self.Jopt, 'P/D', self.PDopt, 'EAR', self.EARopt,'Max Efficiency', \
                       self.maxeff, 'Optimum RPM', self.rpm, 'Thrust [lbs]', self.Thrust,\
                       'Torque [ft lb]', self.Torque )
            dialog1 = wx.MessageBox (results1, 'Results', style = wx.OK )
        elif self.lselect1 == 2:
            results2 ="%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f" \
                     % ('J', self.Jopt, 'P/D', self.PDopt, 'EAR', self.EARopt,'Max Efficiency', \
                       self.maxeff, 'Optimum Diameter', self.D, 'Thrust [lbs]', self.Thrust, \
                       'Torque [ft lb]', self.Torque )
            dialog2 = wx.MessageBox (results2, 'Results', style = wx.OK )
        elif self.lselect1 == 3:
            results3 ="%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f\n\
%15s = %6.4f" \
                     % ('J', self.Jopt, 'P/D', self.PDopt, 'EAR', self.EARopt,'Max Efficiency',\
                       self.maxeff, 'Optimum Diameter', self.D, 'Thrust [lbs]', self.Thrust,\
                       'Torque [ft lb]', self.Torque )
            dialog3 = wx.MessageBox (results3, 'Results', style = wx.OK )

if __name__ == '__main__':
    app = wx.PySimpleApp()
    TextFrame().Show()
    app.MainLoop()

