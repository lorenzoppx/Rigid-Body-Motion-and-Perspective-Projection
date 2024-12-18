import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QWidget, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton,QGroupBox
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from numpy import array
import numpy as np

# Função para plotar gráfico
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

# Função para plotar as setas
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

#Função para realizar translações
def translation_M(D):
  dx, dy, dz = D
  T = np.array([[1,0,0,dx],[0,1,0,dy],[0,0,1,dz],[0,0,0,1]])
  return T

#Função para realizar rotação em X
def rotation_x_M(angle):
  angle = np.deg2rad(-angle)
  Rx = np.array([[1,0,0,0],[0,np.cos(angle),np.sin(angle),0],[0,-np.sin(angle),np.cos(angle),0],[0,0,0,1]])
  return Rx

#Função para realizar rotação em Y
def rotation_y_M(angle):
  angle = np.deg2rad(-angle)
  Ry = np.array([[np.cos(angle),0,-np.sin(angle),0],[0,1,0,0],[np.sin(angle),0,np.cos(angle),0],[0,0,0,1]])
  return Ry

#Função para realizar rotação em Z
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

    # Função que seta a origem, a câmera e define sua posição inicial
    def set_cam(self):
        # base vector values
        e1 = np.array([[1],[0],[0],[0]]) # X
        e2 = np.array([[0],[1],[0],[0]]) # Y
        e3 = np.array([[0],[0],[1],[0]]) # Z
        self.base = np.hstack((e1,e2,e3))
        print ('Cartesian base: \n',self.base)
        
        # Ponto de origem
        self.point = np.array([[0],[0],[0],[1]])
        self.cam0 = np.hstack((self.base,self.point))
        self.cam = self.cam0

        # Define uma posição inicial da câmera para boa visualização do objeto
        self.cam = rotation_y_M(-90)@rotation_z_M(90)@translation_M([0,8,-25])@self.cam
        self.world = self.cam
        
        print ('Origin: \n',self.point)
        print ('cam: \n',self.cam)

        self.world0 = self.world

    # Função que carrega o STL e seta o objeto 3d
    def set_rabbit(self):
        from stl import mesh
        print("Building -> Rabbit")

        # Load the STL files and add the vectors to the plot
        self.rabbit = mesh.Mesh.from_file('coelho.stl')
        
        def scale(k):
            M = (np.eye(4)) * k
            M[-1,:] = [0,0,0,1]
            return M
        # Define scale matrix
        S = scale(0.1)

        x = self.rabbit.x.flatten()
        y = self.rabbit.y.flatten()
        z = self.rabbit.z.flatten()
        
        # Create the 3D object from the x,y,z coordinates and add the additional array of ones to 
        # represent the object using homogeneous coordinates
        self.rabbit = np.array([x.T, y.T, z.T, np.ones(x.size)])
        
        # Extract vectors of mesh
        self.rabbit = self.rabbit

        # Apply Scale
        self.rabbit = S@self.rabbit
    
        return

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
        self.canvas_layout = None
        self.set_rabbit()

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
        line_edit_widget2 = self.create_cam_widget("Ref camera")
        line_edit_widget3 = self.create_intrinsic_widget("Params instr")
        
        self.world_widget = line_edit_widget1
        self.cam_widget = line_edit_widget2
        self.intrinsic_widget = line_edit_widget3

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
        # Criar um widget para exibir os gráficos do Matplotlib
        canvas_widget = QWidget()
        self.canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(self.canvas_layout)

        # Criar um objeto FigureCanvas para exibir o gráfico 2D
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title("Imagem")
        self.canvas1 = FigureCanvas(self.fig1)

        ##### Acertando limites do eixo X
        self.ax1.set_xlim([0,1920])
        
        ##### Acertando limites do eixo Y
        self.ax1.set_ylim([1080,0])

        ##### Criando a função de projeção 
        object_2d = self.projection_2d()
        self.ax1.plot(object_2d[0,:], object_2d[1,:])

        ##### Plotando o object_2d que retornou da projeção
        self.ax1.grid('True')
        self.ax1.set_aspect('equal')
        self.canvas_layout.addWidget(self.canvas1)

        # Criar um objeto FigureCanvas para exibir o gráfico 3D
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        
        ##### Plotando o objeto 3D e os referenciais da câmera e do mundo
        self.ax2 = set_plot(self.ax2,lim=[-15,20])
        
        # Plotando o objeto 3D
        self.ax2.plot3D(self.rabbit[0,:], self.rabbit[1,:], self.rabbit[2,:], 'red')
        
        # Plotando o referencial do mundo
        draw_arrows(self.point,self.base,self.ax2,3.5)

        # Plotando o referencial da câmera
        draw_arrows(self.cam[:,-1],self.cam[:,0:3],self.ax2,3.5)

        self.canvas2 = FigureCanvas(self.fig2)
        self.canvas2.draw()
        self.canvas_layout.addWidget(self.canvas2)
        
        # Retornar o widget de canvas
        return canvas_widget

    def convert_line_edits_intr(self,line_edits):
        labels = ['n_pixels_base:', 'n_pixels_altura:', 'ccd_x:', 'ccd_y:', 'dist_focal:', 'sθ:']
        line_edits_result = []
        for i in range(0,len(labels)):
            if line_edits[i].text()!='':
                line_edits_result.append(float(line_edits[i].text()))
            else:
                line_edits_result.append(0)
            print(labels[i],':',line_edits_result[i])

        if line_edits_result[0]!=0:
            self.n_pixels_base = line_edits_result[0]
        if line_edits_result[1]!=0:
            self.n_pixels_altura = line_edits_result[1]
        if line_edits_result[2]!=0: 
            self.ccd_x = line_edits_result[2]
        if line_edits_result[3]!=0:
            self.ccd_y = line_edits_result[3]
        if line_edits_result[4]!=0:
            self.dist_focal = line_edits_result[4]
        if line_edits_result[5]!=0:
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
    
    # Função para atualizar parâmetros intrínsecos
    def update_params_intrinsc(self, line_edits):
        print('-Update intrinsc-')
        self.convert_line_edits_intr(line_edits)
        M = self.generate_intrinsic_params_matrix()
        print(M)
        self.update_canvas()
        return 

    # Função para atualizar parâmetros extrinsecos - referencial mundo 
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
        self.cam = translation_M([self.X_move,self.Y_move,self.Z_move])@self.M@self.cam2world@self.cam
        print('cam:',self.cam)
        self.update_canvas()
        return

    # Função para atualizar parâmetros extrinsecos - referencial câmera 
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
    
    # Faz a projeção do objeto em relação a posição da câmera
    def projection_2d(self):
        self.K_pi = np.eye(3)
        self.K_pi = np.hstack((self.K_pi,np.zeros([3,1])))
        self.G = np.linalg.inv(self.cam)
        self.generate_intrinsic_params_matrix()

        # Matriz de projeção
        matrix_cam_view = self.K@self.K_pi@self.G
        print("cam0:",matrix_cam_view )

        # Projeção e criação da imagem
        self.object_cam_view = matrix_cam_view@self.rabbit
        print("cam0:",self.object_cam_view )

        # Preparação das coordenadas na forma cartesiana
        self.object_cam_view[0,:] = self.object_cam_view[0,:]/self.object_cam_view[2,:]
        self.object_cam_view[1,:] = self.object_cam_view[1,:]/self.object_cam_view[2,:]
        self.object_cam_view[2,:] = 1

        print("cam0:",self.object_cam_view)

        return self.object_cam_view
    
    # Função que retorna a matrix de parâmetros intrinsecos
    def generate_intrinsic_params_matrix(self):
        f = self.dist_focal
        fsx = f*self.n_pixels_base/self.ccd_x
        fsy = f*self.n_pixels_altura/self.ccd_y
        ox = self.n_pixels_altura/2
        oy = self.n_pixels_base/2
        # Constroi matriz de parâmetros intrinsecos
        self.K = np.array([[fsx,0,0],[self.sθ,fsy,0],[ox,oy,1]])
        self.K = self.K.T
        return self.K
    
    # Função para atualizar visualização dos gráficos
    def update_canvas(self):
        self.canvas1.flush_events()
        self.canvas2.flush_events()
        self.ax1.cla()
        self.ax2.cla()

        self.ax1.set_title("Imagem")

        ##### Acertando limites do eixo X
        self.ax1.set_xlim([0,self.n_pixels_base])

        ##### Acertando limites do eixo Y
        self.ax1.set_ylim([self.n_pixels_altura,0])
        
        ##### Função de projeção 
        object_2d = self.projection_2d()
        self.ax1.plot(object_2d[0,:], object_2d[1,:])
        self.canvas1.draw()

        ##### Plotando o object_2d que retornou da projeção

        self.ax1.grid('True')
        self.ax1.set_aspect('equal')
        
        ##### Plotando o objeto 3D e os referenciais da câmera e do mundo
        self.ax2 = set_plot(self.ax2,lim=[-15,20])

        # Plotando o objeto 3D
        self.ax2.plot3D(self.rabbit[0,:], self.rabbit[1,:], self.rabbit[2,:], 'red')
        
        # Plotando referencial do mundo
        draw_arrows(self.point,self.base,self.ax2,3.5)
        
        # Plotando referencial da câmera
        draw_arrows(self.cam[:,-1],self.cam[:,0:3],self.ax2,3.5)

        self.canvas2.draw()
        
        # Limpa os campos de digitação
        self.reset_params()

        # Retornar o widget de canvas
        return

    # Função para resetar visualização dos gráficos - posiciona câmera na origem
    def reset_canvas(self):
        print('-Reset canvas-')
        # Posiciona câmera na origem do mundo
        self.cam = self.cam0
        self.update_canvas()
        return
    
    def reset_params(self):
        # Lista de todos os widgets que contêm os campos de texto
        widgets_to_clear = [self.world_widget, self.cam_widget, self.intrinsic_widget]

        for widget in widgets_to_clear:
            # Encontrar todas as QLineEdit no widget atual
            line_edits = widget.findChildren(QLineEdit)
            for line_edit in line_edits:
                line_edit.clear()
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
