import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
import numpy as np


def set_plot(ax=None,figure = None,lim=[-2,2]):
    if figure ==None:
        figure = plt.figure(figsize=(8,8))
    if ax==None:
        ax = plt.axes(projection='3d')

    ax.set_title("Camera reference")
    ax.set_xlim(lim)
    ax.set_xlabel("x axis")
    ax.set_ylim(lim)
    ax.set_ylabel("y axis")
    ax.set_zlim(lim)
    ax.set_zlabel("z axis")
    return ax

#adding quivers to the plot
def draw_arrows(point,base,axis,length=1.5):
    # The object base is a matrix, where each column represents the vector
    # of one of the axis, written in homogeneous coordinates (ax,ay,az,0)

    # Plot vector of x-axis
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',  length=length)
    # Plot vector of y-axis
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',  length=length)
    # Plot vector of z-axis
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)

    return axis


###### Crie suas funções de translação, rotação, criação de referenciais, plotagem de setas e qualquer outra função que precisar

#Functions for homogenous coordinates
def translation_M(D):
  dx, dy, dz = D
  T = np.array([[1,0,0,dx],[0,1,0,dy],[0,0,1,dz],[0,0,0,1]])
  return T

def rotation_x_M(angle):
  angle = np.deg2rad(-angle)
  Rx = np.array([[1,0,0,0],[0,np.cos(angle),np.sin(angle),0],[0,-np.sin(angle),np.cos(angle),0],[0,0,0,1]])
  return Rx

def rotation_y_M(angle):
  angle = np.deg2rad(-angle)
  Ry = np.array([[np.cos(angle),0,-np.sin(angle),0],[0,1,0,0],[np.sin(angle),0,np.cos(angle),0],[0,0,0,1]])
  return Ry

def rotation_z_M(angle):
  angle = np.deg2rad(-angle)
  Rz = np.array([[np.cos(angle),np.sin(angle),0,0],[-np.sin(angle),np.cos(angle),0,0],[0,0,1,0],[0,0,0,1]])
  return Rz



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #definindo as variaveis
        self.set_variables()
        #Ajustando a tela    
        self.setWindowTitle("Grid Layout")
        self.setGeometry(100, 100,1280 , 720)
        self.setup_ui()

    def set_cam(self):
        # base vector values
        e1 = np.array([[1],[0],[0],[0]]) # X
        e2 = np.array([[0],[1],[0],[0]]) # Y
        e3 = np.array([[0],[0],[1],[0]]) # Z
        self.base = np.hstack((e1,e2,e3))
        print ('Cartesian base: \n',self.base)
        #origin point
        self.point = np.array([[0],[0],[0],[1]])
        self.cam = np.hstack((self.base,self.point))
        self.world = self.cam
        print ('Origin: \n',self.point)
        print ('cam: \n',self.cam)

        self.world0 = self.world
        self.cam0 = self.cam

    def set_elephant(self):
        from stl import mesh
        print("Building -> Elephant")

        # Load the STL files and add the vectors to the plot
        self.elephant = mesh.Mesh.from_file('coelho.stl')
        
        def scale(k):
            M = (np.eye(4)) * k
            M[-1,:] = [0,0,0,1]
            return M
        # Define scale matrix
        S = scale(0.1)

        x = self.elephant.x.flatten()
        y = self.elephant.y.flatten()
        z = self.elephant.z.flatten()
        
        # Create the 3D object from the x,y,z coordinates and add the additional array of ones to 
        # represent the object using homogeneous coordinates
        self.elephant = np.array([x.T, y.T, z.T, np.ones(x.size)])
        
        # Extract vectors of mesh
        self.elephant = self.elephant

        # Apply Scale
        self.elephant = S@self.elephant
    
        return


    #Creating a house
    def set_house(self):
        self.house = np.array([[0,         0,         0],
                [0,  -10.0000,         0],
                [0, -10.0000,   12.0000],
                [0,  -10.4000,   11.5000],
                [0,   -5.0000,   16.0000],
                [0,         0,   12.0000],
                [0,    0.5000,   11.4000],
                [0,         0,   12.0000],
                [0,         0,         0],
        [-12.0000,         0,         0],
        [-12.0000,   -5.0000,         0],
        [-12.0000,  -10.0000,         0],
                [0,  -10.0000,         0],
                [0,  -10.0000,   12.0000],
        [-12.0000,  -10.0000,   12.0000],
        [-12.0000,         0,   12.0000],
                [0,         0,   12.0000],
                [0,  -10.0000,   12.0000],
                [0,  -10.5000,   11.4000],
        [-12.0000,  -10.5000,   11.4000],
        [-12.0000,  -10.0000,   12.0000],
        [-12.0000,   -5.0000,   16.0000],
                [0,   -5.0000,   16.0000],
                [0,    0.5000,   11.4000],
        [-12.0000,    0.5000,   11.4000],
        [-12.0000,         0,   12.0000],
        [-12.0000,   -5.0000,   16.0000],
        [-12.0000,  -10.0000,   12.0000],
        [-12.0000,  -10.0000,         0],
        [-12.0000,   -5.0000,         0],
        [-12.0000,         0,         0],
        [-12.0000,         0,   12.0000],
        [-12.0000,         0,         0]])

        self.house = np.transpose(self.house)

        #add a vector of ones to the house matrix to represent the house in homogeneous coordinates
        self.house = np.vstack([self.house, np.ones(np.size(self.house,1))])

    def set_variables(self):
        self.objeto_original = [] #modificar
        self.objeto = self.objeto_original
        self.cam_original = [] #modificar
        self.cam = [] #modificar
        self.px_base = 1280  #modificar
        self.px_altura = 720 #modificar
        self.dist_foc = 50 #modificar
        self.stheta = 0 #modificar
        self.ox = self.px_base/2 #modificar
        self.oy = self.px_altura/2 #modificar
        self.ccd = [36,24] #modificar
        self.projection_matrix = [] #modificar
        self.set_cam()
        #self.world = translation_M([1,1,1])@self.world
        self.canvas_layout = None
        self.set_house()
        self.set_elephant()

        self.n_pixels_base = 1920
        self.n_pixels_altura = 1920
        self.ccd_x = 36
        self.ccd_y = 36
        self.dist_focal = 25
        self.sθ = 0


    def setup_ui(self):
        # Criar o layout de grade
        grid_layout = QGridLayout()

        # Criar os widgets
        line_edit_widget1 = self.create_world_widget("Ref mundo")
        line_edit_widget2  = self.create_cam_widget("Ref camera")
        line_edit_widget3  = self.create_intrinsic_widget("Params instr")

        self.canvas = self.create_matplotlib_canvas()

        # Adicionar os widgets ao layout de grade
        grid_layout.addWidget(line_edit_widget1, 0, 0)
        grid_layout.addWidget(line_edit_widget2, 0, 1)
        grid_layout.addWidget(line_edit_widget3, 0, 2)
        grid_layout.addWidget(self.canvas, 1, 0, 1, 3)

        # Criar um widget para agrupar o botão de reset
        reset_widget = QWidget()
        reset_layout = QHBoxLayout()
        reset_widget.setLayout(reset_layout)

        # Criar o botão de reset vermelho
        reset_button = QPushButton("Reset")
        reset_button.setFixedSize(50, 30)  # Define um tamanho fixo para o botão (largura: 50 pixels, altura: 30 pixels)
        style_sheet = """
            QPushButton {
                color : black ;
                background: rgba(255, 127, 130,255);
                font: inherit;
                border-radius: 5px;
                line-height: 1;
            }
        """
        reset_button.setStyleSheet(style_sheet)
        reset_button.clicked.connect(self.reset_canvas)

        # Adicionar o botão de reset ao layout
        reset_layout.addWidget(reset_button)

        # Adicionar o widget de reset ao layout de grade
        grid_layout.addWidget(reset_widget, 2, 0, 1, 3)

        # Criar um widget central e definir o layout de grade como seu layout
        central_widget = QWidget()
        central_widget.setLayout(grid_layout)
        
        # Definir o widget central na janela principal
        self.setCentralWidget(central_widget)

    def create_intrinsic_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['n_pixels_base:', 'n_pixels_altura:', 'ccd_x:', 'ccd_y:', 'dist_focal:', 'sθ:']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_params_intrinsc ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_params_intrinsc(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget
    
    def create_world_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_world ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_world(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget

    def create_cam_widget(self, title):
        # Criar um widget para agrupar os QLineEdit
        line_edit_widget = QGroupBox(title)
        line_edit_layout = QVBoxLayout()
        line_edit_widget.setLayout(line_edit_layout)

        # Criar um layout de grade para dividir os QLineEdit em 3 colunas
        grid_layout = QGridLayout()

        line_edits = []
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit

        # Adicionar widgets QLineEdit com caixa de texto ao layout de grade
        for i in range(1, 7):
            line_edit = QLineEdit()
            label = QLabel(labels[i-1])
            validator = QDoubleValidator()  # Validador numérico
            line_edit.setValidator(validator)  # Aplicar o validador ao QLineEdit
            grid_layout.addWidget(label, (i-1)//2, 2*((i-1)%2))
            grid_layout.addWidget(line_edit, (i-1)//2, 2*((i-1)%2) + 1)
            line_edits.append(line_edit)

        # Criar o botão de atualização
        update_button = QPushButton("Atualizar")

        ##### Você deverá criar, no espaço reservado ao final, a função self.update_cam ou outra que você queira 
        # Conectar a função de atualização aos sinais de clique do botão
        update_button.clicked.connect(lambda: self.update_cam(line_edits))

        # Adicionar os widgets ao layout do widget line_edit_widget
        line_edit_layout.addLayout(grid_layout)
        line_edit_layout.addWidget(update_button)

        # Retornar o widget e a lista de caixas de texto
        return line_edit_widget

    def create_matplotlib_canvas(self):
        #if self.canvas_layout is not None:
        #    self.canvas_layout.deleteLater()
        # Criar um widget para exibir os gráficos do Matplotlib
        canvas_widget = QWidget()
        self.canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(self.canvas_layout)

        # Criar um objeto FigureCanvas para exibir o gráfico 2D
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title("Imagem")
        self.canvas1 = FigureCanvas(self.fig1)

        # Acerte os limites do eixo X
        self.ax1.set_xlim([0,1920])
        # Acerte os limites do eixo Y
        # Para inverter, basta colocar o valor máximo primeiro e o valor mínimo depois
        self.ax1.set_ylim([1080,0])

        ##### Você deverá criar a função de projeção 
        object_2d = self.projection_2d()
        self.ax1.plot(object_2d[0,:], object_2d[1,:])

        ##### Falta plotar o object_2d que retornou da projeção
          
        self.ax1.grid('True')
        self.ax1.set_aspect('equal')
        self.canvas_layout.addWidget(self.canvas1)

        # Criar um objeto FigureCanvas para exibir o gráfico 3D
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
        ##### Falta plotar o seu objeto 3D e os referenciais da câmera e do mundo

        self.ax2 = set_plot(self.ax2,lim=[-15,20])
        #draw_arrows(self.point,self.base,self.ax2)
        self.ax2.plot3D(self.elephant[0,:], self.elephant[1,:], self.elephant[2,:], 'red')
        # Plotando a quina da casa que está em (0,0,0) para servir de referência
        #self.ax2.scatter(self.house[0,0], self.house[1,0], self.house[2,0],'b')
        # Plote a câmera também - adicione o código abaixo

        #draw_arrows(self.world[:,-1],self.world[:,0:3],self.ax2)
        #draw_arrows(self.point,self.base,self.ax2,1.5)
        draw_arrows(self.cam[:,-1],self.cam[:,0:3],self.ax2,3.5)

        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.draw()

        self.canvas_layout.addWidget(self.canvas2)
        
        #canvas_widget.update()
        #canvas_widget.updatesEnabled()
        # Retornar o widget de canvas
        return canvas_widget


    ##### Você deverá criar as suas funções aqui
    def convert_line_edits_intr(self,line_edits):
        labels = ['n_pixels_base:', 'n_pixels_altura:', 'ccd_x:', 'ccd_y:', 'dist_focal:', 'sθ:']
        line_edits_result = []
        for i in range(0,len(labels)):
            if line_edits[i].text()!='':
                line_edits_result.append(float(line_edits[i].text()))
            else:
                line_edits_result.append(0)
            print(labels[i],':',line_edits_result[i])

        self.n_pixels_base = line_edits_result[0]
        self.n_pixels_altura = line_edits_result[1]
        self.ccd_x = line_edits_result[2]
        self.ccd_y = line_edits_result[3]
        self.dist_focal = line_edits_result[4]
        self.sθ = line_edits_result[5]

        return line_edits_result
    
    def convert_line_edits_ref(self,line_edits):
        labels = ['X(move):', 'X(angle):', 'Y(move):', 'Y(angle):', 'Z(move):', 'Z(angle):']  # Texto a ser exibido antes de cada QLineEdit
        line_edits_result = []
        for i in range(0,len(labels)):
            if line_edits[i].text()!='':
                line_edits_result.append(float(line_edits[i].text()))
            else:
                line_edits_result.append(0)
            print(labels[i],':',line_edits_result[i])

        self.X_move = line_edits_result[0]
        self.X_angle = line_edits_result[1]
        self.Y_move = line_edits_result[2]
        self.Y_angle = line_edits_result[3]
        self.Z_move = line_edits_result[4]
        self.Z_angle = line_edits_result[5]

        return line_edits_result
    
    def update_params_intrinsc(self, line_edits):
        print('-Update intrinsc-')
        self.convert_line_edits_intr(line_edits)
        M = self.generate_intrinsic_params_matrix()
        print(M)
        self.update_canvas()
        return 

    def update_world(self,line_edits):
        print('-Update world-')
        self.convert_line_edits_ref(line_edits)
        self.world2cam = np.linalg.inv(self.cam)
        self.cam2world = self.cam
        self.cam = self.world2cam@self.cam
        self.M = self.cam0
        if self.X_angle!=0:
            self.M = rotation_x_M(self.X_angle)@self.M
        if self.Y_angle!=0:
            self.M = rotation_y_M(self.Y_angle)@self.M
        if self.Z_angle!=0:
            self.M = rotation_z_M(self.Z_angle)@self.M
        print('cam2_world:',self.cam2world)
        #self.Tcam = translation_M(self.cam2world[:-1,-1])
        #self.cam2world[:-1,-1] = 0
        #self.Mcam = self.cam2world
        print('cam2_world:',self.cam2world)
        #self.cam = translation_M([self.X_move,self.Y_move,self.Z_move])@self.M@self.Mcam@self.Tcam@self.cam
        self.cam = translation_M([self.X_move,self.Y_move,self.Z_move])@self.M@self.cam2world@self.cam
        print('cam:',self.cam)
        self.update_canvas()
        return

    def update_cam(self,line_edits):
        print('-Update cam-')
        self.convert_line_edits_ref(line_edits)
        self.world2cam = np.linalg.inv(self.cam)
        self.cam2world = self.cam
        self.cam = self.world2cam@self.cam
        self.cam = translation_M([self.X_move,self.Y_move,self.Z_move])@self.cam
        if self.X_angle!=0:
            self.cam = rotation_x_M(self.X_angle)@self.cam
        if self.Y_angle!=0:
            self.cam = rotation_y_M(self.Y_angle)@self.cam
        if self.Z_angle!=0:
            self.cam = rotation_z_M(self.Z_angle)@self.cam
        self.cam = self.cam2world@self.cam
        print('cam:',self.cam)
        self.update_canvas()
        return 
    
    def projection_2d(self):
        self.K_pi = np.eye(3)
        self.K_pi = np.hstack((self.K_pi,np.zeros([3,1])))
        self.G = np.linalg.inv(self.cam)
        self.generate_intrinsic_params_matrix()

        # Matriz de projeção
        matrix_cam_view = self.K@self.K_pi@self.G
        print("cam0:",matrix_cam_view )

        # Projeção e criação da imagem

        self.house_cam_view = matrix_cam_view@self.elephant
        print("cam0:",self.house_cam_view )

        # Preparação das coordenadas na forma cartesiana

        self.house_cam_view[0,:] = self.house_cam_view[0,:]/self.house_cam_view[2,:]
        self.house_cam_view[1,:] = self.house_cam_view[1,:]/self.house_cam_view[2,:]
        self.house_cam_view[2,:] = 1

        print("cam0:",self.house_cam_view)

        return self.house_cam_view
    
    def generate_intrinsic_params_matrix(self):
        # Distancia focal em pixel
        f = self.dist_focal
        fsx = f*self.n_pixels_base/self.ccd_x
        fsy = f*self.n_pixels_altura/self.ccd_y
        ox = self.n_pixels_altura/2
        oy = self.n_pixels_base/2
        # matriz de parametros intrinsecos
        self.K = np.array([[fsx,0,0],[self.sθ,fsy,0],[ox,oy,1]])
        self.K = self.K.T
        return self.K
    
    def update_canvas(self):
        # Criar um objeto FigureCanvas para exibir o gráfico 2D
        self.canvas1.flush_events()
        self.canvas2.flush_events()
        self.ax1.cla()
        self.ax2.cla()

        self.ax1.set_title("Imagem")
        #self.canvas1 = FigureCanvas(self.fig1)
        # Acerte os limites do eixo X
        self.ax1.set_xlim([0,self.n_pixels_base])
        # Acerte os limites do eixo Y
        # Para inverter, basta colocar o valor máximo primeiro e o valor mínimo depois
        self.ax1.set_ylim([self.n_pixels_altura,0])
        
        ##### Você deverá criar a função de projeção 
        object_2d = self.projection_2d()
        self.ax1.plot(object_2d[0,:], object_2d[1,:])
        self.canvas1.draw()
        ##### Falta plotar o object_2d que retornou da projeção
          
        self.ax1.grid('True')
        self.ax1.set_aspect('equal')
        
        ##### Falta plotar o seu objeto 3D e os referenciais da câmera e do mundo

        self.ax2 = set_plot(self.ax2,lim=[-15,20])

        #draw_arrows(self.point,self.base,self.ax2)
        self.ax2.plot3D(self.elephant[0,:], self.elephant[1,:], self.elephant[2,:], 'red')
        # Plotando a quina da casa que está em (0,0,0) para servir de referência
        #self.ax2.scatter(self.house[0,0], self.house[1,0], self.house[2,0],'b')
        # Plote a câmera também - adicione o código abaixo

        #draw_arrows(self.world[:,-1],self.world[:,0:3],self.ax2,3.5)
        #draw_arrows(self.point,self.base,self.ax2,3.5)
        draw_arrows(self.cam[:,-1],self.cam[:,0:3],self.ax2,3.5)

        #self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.draw()
        
        #canvas_widget.update()
        #canvas_widget.updatesEnabled()
        # Retornar o widget de canvas
        return
    
    def reset_canvas(self):
        print('-Reset canvas-')
        self.cam = self.cam0
        #self.ax2.remove()
        #self.fig2.clear()
        #self.fig1.clear()
        self.update_canvas()
        return
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
