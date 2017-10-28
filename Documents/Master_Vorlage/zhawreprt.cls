%%
%% This is file `zhawreprt.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% zhawreprt.dtx  (with options: `class')
%% 
%% IMPORTANT NOTICE:
%% 
%% For the copyright see the source file.
%% 
%% Any modified versions of this file must be renamed
%% with new filenames distinct from zhawreprt.cls.
%% 
%% For distribution of the original source see the terms
%% for copying and modification in the file zhawreprt.dtx.
%% 
%% This generated file may be distributed as long as the
%% original source files, as listed above, are part of the
%% same distribution. (The sources need not necessarily be
%% in the same archive or directory.)
%% File: zhawreprt.dtx  Copyright (C) 2016--2021  Raphael Emberger
%% $Id: zhawreprt.dtx,v 1.0 2016/08/23 10:51:00 m Rel $
%% $Version: 1.0 $
\ProvidesClass{zhawreprt}[2016/08/23 v1.0 ZHAW style class]
\newif\iffinal\finalfalse
\DeclareOption{final}{\finaltrue}
\newif\if@german\@germanfalse
\DeclareOption{german}{\@germantrue\PassOptionsToClass{german}{scrreprt}}
\DeclareOption{ngerman}{\@germantrue\PassOptionsToClass{ngerman}{scrreprt}}
\DeclareOption{english}{\@germanfalse\PassOptionsToClass{english}{scrreprt}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrreprt}}
\ProcessOptions\relax
\LoadClass{scrreprt}

\RequirePackage[scaled]{uarial}
	\renewcommand*\familydefault{\sfdefault}
\RequirePackage[T1]{fontenc}
\RequirePackage{babel}
\RequirePackage{polyglossia}
\RequirePackage{graphicx}
\RequirePackage{geometry}
	\geometry{a4paper, top=30mm, left=30mm, right=30mm, bottom=30mm, headsep=10mm, footskip=10mm}
\RequirePackage[absolute]{textpos}
	\setlength{\TPHorizModule}{30mm}
	\setlength{\TPVertModule}{\TPHorizModule}
	\textblockorigin{10mm}{10mm}
	\setlength{\parindent}{0pt}
\RequirePackage{pdflscape}
\RequirePackage{pdfpages}
\RequirePackage{makeidx}
\RequirePackage{url}
\RequirePackage{wallpaper}
\RequirePackage{float}
\RequirePackage{scrhack}
\RequirePackage[headsepline,footsepline,markcase=noupper]{scrlayer-scrpage}
\RequirePackage{pdftexcmds}




\setlength{\columnsep}{25pt}
\setlength{\columnseprule}{0.4pt}


\def\theprojecttype{PA}\newcommand{\projecttype}[1]{
\ifnum\pdf@strcmp{\unexpanded{#1}}{PA}=0 %
\def\theprojecttype{PA}
\if@german\def\zhaw@Project{Projektarbeit}\def\zhaw@project{Projektarbeit}\else\def\zhaw@Project{Project Work}\def\zhaw@project{project work}\fi
\else\ifnum\pdf@strcmp{\unexpanded{#1}}{BA}=0 %
\if@german\def\zhaw@Project{Bachelorarbeit}\def\zhaw@project{Bachelorarbeit}\else\def\zhaw@Project{Bachelor's Thesis}\def\zhaw@project{Bachelor's thesis}\fi
\else
\PackageError{zhawreprt}{I don't know what kind of project type you want. I know `PA' and `BA' but what is `#1' supposed to mean?}
\fi\fi}
\def\thelogofilename{images/logos/SoE/de/de-soe-cmyk.png}\newcommand{\logofilename}[1]{\def\thelogofilename{#1}}
\def\zhaw@shorttitle{}\newcommand{\shorttitle}[1]{\def\zhaw@shorttitle{#1}}
\def\zhaw@major{}\newcommand{\major}[1]{\def\zhaw@major{#1}}
\def\zhaw@authors{}\newcommand{\authors}[1]{\def\zhaw@authors{#1}}
\def\zhaw@mainreferee{}\newcommand{\mainreferee}[1]{\def\zhaw@mainreferee{#1}}
\def\zhaw@referee{}\newcommand{\referee}[1]{\def\zhaw@referee{#1}}
\def\zhaw@industrypartner{}\newcommand{\industrypartner}[1]{\def\zhaw@industrypartner{#1}}
\def\zhaw@extreferee{}\newcommand{\extreferee}[1]{\def\zhaw@extreferee{#1}}
\def\zhaw@date{}\newcommand{\setdate}[1]{\def\zhaw@date{#1}}
\newcommand{\renameindex}{\if@german\renewcommand{\indexname}{Stichwortverzeichnis}\else\renewcommand{\indexname}{Index}\fi}


\setlength{\headheight}{50pt}
\setlength{\topmargin}{-55pt}
\setlength{\headsep}{10pt}
\setlength{\voffset}{15pt}
\setlength{\marginparwidth}{55pt}
\setlength{\footskip}{35pt}
\setlength{\hoffset}{5pt}
\renewcommand*{\sls@headabove@linethickness}{0pt}
\renewcommand*{\sls@headbelow@linethickness}{1pt}
\renewcommand*{\sls@footabove@linethickness}{1pt}
\renewcommand*{\sls@footbelow@linethickness}{0pt}
\renewcommand*\chapterpagestyle{scrheadings}
\renewcommand*\partpagestyle{scrheadings}
\automark[section]{chapter}
\renewcommand\chaptermarkformat{}
\renewcommand\sectionmarkformat{}
\clearpairofpagestyles
\lohead{\includegraphics[height={\headheight-4pt}]{\thelogofilename}}
\cohead{\leftmark}
\rohead{\zhaw@shorttitle}
\lehead{\includegraphics[height={\headheight-4pt}]{\thelogofilename}}
\cehead{\leftmark}
\rehead{\zhaw@shorttitle}
\lofoot{}
\cofoot{}
\rofoot{\pagemark}
\lefoot{\pagemark}
\cefoot{}
\refoot{}
\pagestyle{scrheadings}


\newpairofpagestyles[scrheadings]{declaration}{
\lohead{}
\cohead{\leftmark}
\rohead{\zhaw@shorttitle}
\lehead{}
\cehead{\leftmark}
\rehead{\zhaw@shorttitle}
\lofoot{}
\cofoot{}
\rofoot{}
\lefoot{}
\cefoot{}
\refoot{}
}
\DeclareNewLayer[
  background,
  topmargin,
  oddpage,
  contents={\tikz[remember picture,overlay]{\node[anchor=north west,inner sep=0pt] (zhaw) at (1,-0.7) {\includegraphics[height=2.7cm]{\thelogofilename}};}}
]{top.odd}
\DeclareNewLayer[
  clone=top.odd,
  evenpage
]{top.even}
\DeclareNewLayer[
  background,
  bottommargin,
  oddpage,
  contents={\color{blue!70!green!80!white}\footnotesize\normalfont\qquad\textbf{\if@german
  Zürcher Fachhochschule
  \else
  Zurich University of Applied Sciences
  \fi}}
]{bottom.odd}
\DeclareNewLayer[
  clone=bottom.odd,
  evenpage
]{bottom.even}
\AddLayersToPageStyle{declaration}{top.odd,top.even,bottom.odd,bottom.even}


\renewcommand*\maketitle[1][1]{%
  \expandafter\ifnum \csname scr@v@3.12\endcsname>\scr@compatibility\relax
  \else
    \def\and{%
      \end{tabular}%
      \hskip 1em \@plus.17fil%
      \begin{tabular}[t]{c}%
    }%
  \fi
  \if@titlepage
    \begin{titlepage}
    \setcounter{page}{#1}
	\ThisTileWallPaper{\paperwidth}{\paperheight}{images/logos/SoE.pdf}%
	\hfill
	\begin{minipage}[b]{0.91\textwidth}
	\vskip 2.8cm
		{\huge
		\textbf{\underline{\zhaw@project}}\\
		\textbf{\underline{\zhaw@major}}\vskip 0.2cm
		\@title
		\vskip 0.5cm}
		\ifx\zhaw@authors\empty\else
		\begin{minipage}[b][1.5cm][t]{0.27\textwidth}
		\hrule\vskip 0.5cm
			\textbf{Autoren}
		\end{minipage}\hfill
		\begin{minipage}[b][1.5cm][t]{0.7\textwidth}
		\hrule\vskip 0.5cm
			\zhaw@authors
		\end{minipage}
		\fi
		\ifx\zhaw@mainreferee\empty\else
		\begin{minipage}[b][1.5cm][t]{0.27\textwidth}
		\hrule\vskip 0.5cm
			\textbf{Hauptbetreuung}
		\end{minipage}\hfill
		\begin{minipage}[b][1.5cm][t]{0.7\textwidth}
		\hrule\vskip 0.5cm
			\zhaw@mainreferee
		\end{minipage}
		\fi
		\ifx\zhaw@referee\empty\else
		\begin{minipage}[b][1.5cm][t]{0.27\textwidth}
		\hrule\vskip 0.5cm
			\textbf{Nebenbetreuung}
		\end{minipage}\hfill
		\begin{minipage}[b][1.5cm][t]{0.7\textwidth}
		\hrule\vskip 0.5cm
			\zhaw@referee
		\end{minipage}
		\fi
		\ifx\zhaw@industrypartner\empty\else
		\begin{minipage}[b][1.5cm][t]{0.27\textwidth}
		\hrule\vskip 0.5cm
			\textbf{Industriepartner}
		\end{minipage}\hfill
		\begin{minipage}[b][1.5cm][t]{0.7\textwidth}
		\hrule\vskip 0.5cm
			\zhaw@industrypartner
		\end{minipage}
		\fi
		\ifx\zhaw@extreferee\empty\else
		\begin{minipage}[b][1.5cm][t]{0.27\textwidth}
		\hrule\vskip 0.5cm
			\textbf{Externe Betreuung}
		\end{minipage}\hfill
		\begin{minipage}[b][1.5cm][t]{0.7\textwidth}
		\hrule\vskip 0.5cm
			\zhaw@extreferee
		\end{minipage}
		\fi
		\begin{minipage}[b][1.5cm][t]{0.27\textwidth}
		\hrule\vskip 0.5cm
			\textbf{Datum}
		\end{minipage}\hfill
		\begin{minipage}[b][1.5cm][t]{0.7\textwidth}
		\hrule\vskip 0.5cm
			\zhaw@date
		\end{minipage}
	\end{minipage}
	\vskip 0.5cm
	\iffinal\else
	\textcolor{darkgray}{
	Bitte füllen Sie das Titelblatt aus und berücksichtigen Sie Folgendes:\\
	 -> Bitte auf keinen Fall Schriftart und Schriftgrösse ändern. Text soll lediglich überschrieben werden!\\
	 -> Bitte pro Tabellenzeile max. 4 Textzeilen!\vspace{1em}\\
	•	Titel: Fügen Sie Ihren Studiengang direkt nach dem Wort „\zhaw@Project“ ein (max. 2 Zeilen).\\
	•	Titel der Arbeit: Überschreiben Sie den Lauftext mit dem Titel Ihrer Arbeit (max. 4 Zeilen).\\
	•	Autoren: Tragen Sie Ihre Vor- und Nachnamen ein (alphabetisch nach Name).\\
	•	Betreuer: Tragen Sie Ihren Betreuer / Ihre Betreuer ein (alphabetisch nach Name).\\
	•	Ohne Nebenbetreuung, Industriepartner oder externe Betreuung, ganze Tabellenzeile löschen.\\
	•	Am Schluss löschen Sie den ganzen Beschrieb (grau) und speichern das Dokument als pdf. ab.%
	}
	\fi
    \end{titlepage}
  \else
    \par
    \@tempcnta=%
    #1%
    \relax\ifnum\@tempcnta=1\else
      \ClassWarning{\KOMAClassName}{%
        Optional argument of \string\maketitle\space ignored
        at\MessageBreak
        notitlepage-mode%
      }%
    \fi
    \begingroup
      \let\titlepage@restore\relax
      \renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
      \let\@oldmakefnmark\@makefnmark
      \renewcommand*{\@makefnmark}{\rlap\@oldmakefnmark}%
      \next@tdpage
      \if@twocolumn
        \ifnum \col@number=\@ne
          \ifx\@extratitle\@empty\else
            \@makeextratitle
            \next@tdpage
          \fi
          \@maketitle
        \else
          \ifx\@extratitle\@empty\else
            \twocolumn[\@makeextratitle]\next@tdpage
          \fi
          \twocolumn[\@maketitle]%
        \fi
      \else
        \ifx\@extratitle\@empty\else
          \@makeextratitle\next@tdpage
        \fi
        \@maketitle
      \fi
      \ifx\titlepagestyle\@empty\else\thispagestyle{\titlepagestyle}\fi
      \@thanks
    \endgroup
  \fi
  \setcounter{footnote}{0}%
  \expandafter\ifnum \csname scr@v@3.12\endcsname>\scr@compatibility\relax
    \let\thanks\relax
    \let\maketitle\relax
    \let\@maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\@subtitle\@empty
    \global\let\@extratitle\@empty
    \global\let\@titlehead\@empty
    \global\let\@subject\@empty
    \global\let\@publishers\@empty
    \global\let\@uppertitleback\@empty
    \global\let\@lowertitleback\@empty
    \global\let\@dedication\@empty
    \global\let\author\relax
    \global\let\title\relax
    \global\let\extratitle\relax
    \global\let\titlehead\relax
    \global\let\subject\relax
    \global\let\publishers\relax
    \global\let\uppertitleback\relax
    \global\let\lowertitleback\relax
    \global\let\dedication\relax
    \global\let\date\relax
  \fi
  \global\let\and\relax
}%
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}

\newcommand{\makedeclarationoforiginality}{
\newpage
\thispagestyle{declaration}
\vspace*{1.6cm}
\if@german
\section*{Erklärung betreffend das selbstständige Verfassen einer \zhaw@Project\ an der School of Engineering}
\vskip 1.2cm
Mit der Abgabe dieser \zhaw@project\ versichert der/die Studierende, dass er/sie die Arbeit selbständig und ohne fremde Hilfe verfasst hat. (Bei Gruppenarbeiten gelten die Leistungen der übrigen Gruppenmitglieder nicht als fremde Hilfe.)
\vskip 0.5em
Der/die unterzeichnende Studierende erklärt, dass alle zitierten Quellen (auch Internetseiten) im Text oder Anhang korrekt nachgewiesen sind, d.h. dass die Projektarbeit keine Plagiate enthält, also keine Teile, die teilweise oder vollständig aus einem fremden Text oder einer fremden Arbeit unter Vorgabe der eigenen Urheberschaft bzw. ohne Quellenangabe übernommen worden sind.
\vskip 0.8em
\textcolor{red}{Bei Verfehlungen aller Art treten die Paragraphen 39 und 40 (Unredlichkeit und Verfahren bei Unredlichkeit) der ZHAW Prüfungsordnung sowie die Bestimmungen der Disziplinarmassnahmen der Hochschulordnung in Kraft.}
\vskip 2.9cm
\begin{minipage}[t]{0.45\textwidth}
Ort, Datum:\vspace{2em}\\
\vspace{2em}\dotfill
\end{minipage}\hfill
\begin{minipage}[t]{0.45\textwidth}
Unterschriften:\vspace{2em}\\
\vspace{2em}\dotfill\\
\vspace{2em}\dotfill\\
\vspace{2em}\dotfill
\end{minipage}
\vfill
Das Original dieses Formulars ist bei der ZHAW-Version aller abgegebenen Projektarbeiten zu Beginn der Dokumentation nach dem Abstract bzw. dem Management Summary mit Original-Unterschriften und -Datum (keine Kopie) einzufügen.
\else
\section*{DECLARATION OF ORIGINALITY\linebreak\zhaw@Project\ at the School of Engineering}
\vskip 1.2cm
By submitting this \zhaw@project, the undersigned student confirms that this thesis is his/her own work and was written without the help of a third party. (Group works: the performance of the other group members are not considered as third party).
\vskip 0.5em
The student declares that all sources in the text (including Internet pages) and appendices have been correctly disclosed. This means that there has been no plagiarism, i.e. no sections of the Bachelor thesis have been partially or wholly taken from other texts and represented as the student’s own work or included without being correctly referenced.
\vskip 0.8em
\textcolor{red}{Any misconduct will be dealt with according to paragraphs 39 and 40 of the General Academic Regulations for Bachelor’s and Master’s Degree courses at the Zurich University of Applied Sciences (Rahmenprüfungsordnung ZHAW (RPO)) and subject to the provisions for disciplinary action stipulated in the University regulations.}
\vskip 2.9cm
\begin{minipage}[t]{0.45\textwidth}
City, Date:\vspace{2em}\\
\vspace{2em}\dotfill
\end{minipage}\hfill
\begin{minipage}[t]{0.45\textwidth}
Signature:\vspace{2em}\\
\vspace{2em}\dotfill\\
\vspace{2em}\dotfill\\
\vspace{2em}\dotfill
\end{minipage}
\vfill
The original signed and dated document (no copies) must be included after the title sheet in the ZHAW version of all Bachelor thesis submitted.
\fi
\renewcommand*{\sls@headbelow@linethickness}{0pt}
\renewcommand*{\sls@footabove@linethickness}{0pt}
\newpage
\renewcommand*{\sls@headbelow@linethickness}{1pt}
\renewcommand*{\sls@footabove@linethickness}{1pt}}

\newcommand{\notes}[1]{
\iffinal\else\textcolor{darkgray}{\begin{itemize}#1\end{itemize}}\fi}

\endinput
