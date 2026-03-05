package co.edu.demoacademico.service;

import co.edu.demoacademico.exception.BusinessException;
import co.edu.demoacademico.model.Estudiante;
import co.edu.demoacademico.model.Grupo;
import co.edu.demoacademico.model.Matricula;
import co.edu.demoacademico.repository.EstudianteRepository;
import co.edu.demoacademico.repository.GrupoRepository;
import co.edu.demoacademico.repository.MatriculaRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@Transactional
public class MatriculaServiceImpl implements MatriculaService {

    private final EstudianteRepository estudianteRepo;
    private final GrupoRepository grupoRepo;
    private final MatriculaRepository matriculaRepo;

    public MatriculaServiceImpl(EstudianteRepository estudianteRepo,
                                GrupoRepository grupoRepo,
                                MatriculaRepository matriculaRepo) {
        this.estudianteRepo = estudianteRepo;
        this.grupoRepo = grupoRepo;
        this.matriculaRepo = matriculaRepo;
    }

    @Override
    public Matricula matricular(Long estudianteId, Long grupoId) {

        // 1) existencia (usa el NotFound del Lab 02 en los services/handlers en la Parte 2)
        Estudiante estudiante = estudianteRepo.findById(estudianteId)
                .orElseThrow(() -> new co.edu.demoacademico.exception.NotFoundException("Estudiante no encontrado: " + estudianteId));

        Grupo grupo = grupoRepo.findById(grupoId)
                .orElseThrow(() -> new co.edu.demoacademico.exception.NotFoundException("Grupo no encontrado: " + grupoId));

        // 2) no duplicado
        if (matriculaRepo.existsByEstudianteIdAndGrupoId(estudianteId, grupoId)) {
            throw new BusinessException("El estudiante ya está matriculado en ese grupo");
        }

        // 3) cupo
        long matriculados = matriculaRepo.countByGrupoId(grupoId);
        if (matriculados >= grupo.getCupoMax()) {
            throw new BusinessException("El grupo no tiene cupo disponible");
        }

        // 4) guardar
        Matricula m = new Matricula();
        m.setEstudiante(estudiante);
        m.setGrupo(grupo);
        return matriculaRepo.save(m);
    }
}